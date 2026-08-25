"""Talks to any OpenAI-compatible chat-completions server over HTTP — no
vendor SDK required, just httpx. Works against Ollama, vLLM, LM Studio,
llama.cpp's server, TGI, Open WebUI, OpenRouter, or OpenAI itself. chat()
returns the `message` dict with role/content/tool_calls, same shape every
one of those servers already returns.
"""

import json
import os

import httpx

DEFAULT_BASE_URL = os.environ.get("LLM_HOST", "http://localhost:11434")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "")  # set via env, never hardcode


def _normalize_tool_call(call: dict) -> dict:
    if not isinstance(call, dict) or call.get("type") != "function":
        return call

    function = call.get("function")
    if not isinstance(function, dict):
        return call

    args = function.get("arguments")
    if args is None or isinstance(args, str):
        return call

    normalized = function.copy()
    normalized["arguments"] = json.dumps(args, ensure_ascii=False)
    return {**call, "function": normalized}


def _normalize_messages(messages: list) -> list:
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            normalized.append(msg)
            continue
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized.append({
                **msg,
                "tool_calls": [_normalize_tool_call(call) for call in tool_calls],
            })
        else:
            normalized.append(msg)
    return normalized


def _content_from_reasoning(message: dict) -> dict:
    """Recover the answer when a server put it in a reasoning field.

    Reasoning models are commonly served with their output split in two:
    `content` for the answer, `reasoning_content` (llama.cpp, vLLM's reasoning
    parsers, DeepSeek) or `reasoning` for the chain of thought. When the model
    emits only reasoning, `content` comes back null or empty — and a turn that
    the server logged as a completed 51-token response then renders as nothing
    at all. Fall back to whichever reasoning field is populated instead.

    Structured `content` (a list of content parts) is left alone; only a
    missing or blank string is filled in."""
    if not isinstance(message, dict):
        return message
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        return message
    if content and content.strip():
        return message
    for key in ("reasoning_content", "reasoning"):
        text = message.get(key)
        if isinstance(text, str) and text.strip():
            return {**message, "content": text}
    return message


class LLMError(Exception):
    pass


async def chat(
    model: str,
    messages: list,
    tools: list = None,
    format: str = None,
    base_url: str = None,
    api_key: str = None,
    timeout: float = 300.0,
) -> dict:
    """POST /v1/chat/completions (OpenAI-compatible) with stream=false and
    return the `message` dict.

    Response shape is `choices[0].message`, which already has the
    role/content/tool_calls keys the rest of the agent expects.

    If an API key is set (via `api_key`, falling back to $LLM_API_KEY),
    it's sent as `Authorization: Bearer <key>`.

    Raises LLMError on a non-2xx response, a connection failure, or an
    unexpected response shape — callers (agent.py, intent.py) already retry
    on this.
    """
    url = f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/v1/chat/completions"
    payload = {"model": model, "messages": _normalize_messages(messages), "stream": False}
    if tools:
        payload["tools"] = tools
    if format == "json":
        payload["response_format"] = {"type": "json_object"}

    key = api_key or DEFAULT_API_KEY
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        body = e.response.text[:300]
        raise LLMError(f"LLM API returned {e.response.status_code} for {model}: {body}") from e
    except httpx.RequestError as e:
        raise LLMError(
            f"Could not reach the LLM server at {url} ({e}). Is it running?"
        ) from e

    choices = data.get("choices")
    if not choices:
        raise LLMError(f"Unexpected response shape from LLM server (no 'choices' key): {data}")
    message = choices[0].get("message")
    if message is None:
        raise LLMError(f"Unexpected response shape from LLM server (no 'message' key): {data}")
    return _content_from_reasoning(message)


async def list_models(base_url: str = None, api_key: str = None, timeout: float = 15.0) -> list:
    """GET /v1/models (OpenAI-compatible) and return the list of model ids
    the server has available. Used by the /model REPL command.

    Raises LLMError on a non-2xx response, a connection failure, or an
    unexpected response shape — same as chat()."""
    url = f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/v1/models"
    key = api_key or DEFAULT_API_KEY
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        body = e.response.text[:300]
        raise LLMError(f"LLM API returned {e.response.status_code} listing models: {body}") from e
    except httpx.RequestError as e:
        raise LLMError(
            f"Could not reach the LLM server at {url} ({e}). Is it running?"
        ) from e

    items = data.get("data")
    if items is None:
        raise LLMError(f"Unexpected response shape from LLM server (no 'data' key): {data}")
    try:
        return [item["id"] for item in items]
    except (KeyError, TypeError) as e:
        raise LLMError(f"Unexpected response shape from LLM models list (no 'id' key): {data}") from e


async def embed(
    model: str,
    input: list,
    base_url: str = None,
    api_key: str = None,
    timeout: float = 30.0,
) -> list:
    """POST /v1/embeddings (OpenAI-compatible) and return one vector per
    string in `input`, same order. Used for semantic search_tools ranking
    (mcp_client.py) — a plain vector lookup, so it's much cheaper than a
    chat() round-trip.

    Raises LLMError on a non-2xx response, a connection failure, or an
    unexpected response shape — same as chat()."""
    url = f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/v1/embeddings"
    payload = {"model": model, "input": input}

    key = api_key or DEFAULT_API_KEY
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        body = e.response.text[:300]
        raise LLMError(f"LLM API returned {e.response.status_code} for embedding model {model}: {body}") from e
    except httpx.RequestError as e:
        raise LLMError(
            f"Could not reach the LLM server at {url} ({e}). Is it running?"
        ) from e

    items = data.get("data")
    if not items:
        raise LLMError(f"Unexpected response shape from LLM embeddings (no 'data' key): {data}")
    try:
        return [item["embedding"] for item in items]
    except (KeyError, TypeError) as e:
        raise LLMError(f"Unexpected response shape from LLM embeddings (no 'embedding' key): {data}") from e
