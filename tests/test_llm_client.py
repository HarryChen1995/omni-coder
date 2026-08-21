"""llm_client — the OpenAI-compatible HTTP layer.

httpx is mocked with its own MockTransport, so requests are intercepted at
the transport boundary: real payload/URL/header construction and real
response parsing run, but nothing leaves the process.
"""

import json

import httpx
import pytest

from omni import llm_client
from omni.llm_client import (
    LLMError, _normalize_messages, _normalize_tool_call, chat, embed, list_models,
)


_REAL_ASYNC_CLIENT = httpx.AsyncClient  # pinned before any patching


def mock_http(mocker, handler):
    """Route every httpx.AsyncClient through `handler`, and record the
    requests it saw. Safe to call more than once per test."""
    seen = []

    def wrapped(request):
        seen.append(request)
        return handler(request)

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(wrapped)
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    mocker.patch.object(llm_client.httpx, "AsyncClient", factory)
    return seen


def json_reply(payload, status=200):
    return lambda request: httpx.Response(status, json=payload)


CHAT_OK = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}


# ---------------- payload normalization ----------------

def test_normalize_tool_call_stringifies_dict_arguments():
    """Some servers echo `arguments` back as a dict; the API contract is a
    JSON string, so it has to be re-encoded before being sent."""
    call = {"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}
    out = _normalize_tool_call(call)
    assert out["function"]["arguments"] == '{"a": 1}'


def test_normalize_tool_call_leaves_string_arguments_alone():
    call = {"type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
    assert _normalize_tool_call(call) == call


@pytest.mark.parametrize("call", [
    {"type": "other"},                                  # not a function call
    {"type": "function", "function": "not-a-dict"},
    {"type": "function", "function": {"name": "f"}},     # no arguments key
    "not-a-dict",
])
def test_normalize_tool_call_passes_through_odd_shapes(call):
    assert _normalize_tool_call(call) == call


def test_normalize_messages_only_touches_tool_calls():
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [
            {"type": "function", "function": {"name": "f", "arguments": {"k": "v"}}}]},
        "not-a-dict",
    ]
    out = _normalize_messages(msgs)
    assert out[0] == msgs[0] and out[2] == "not-a-dict"
    assert out[1]["tool_calls"][0]["function"]["arguments"] == '{"k": "v"}'


# ---------------- chat ----------------

async def test_chat_returns_message(mocker):
    mock_http(mocker, json_reply(CHAT_OK))
    assert await chat("m", [{"role": "user", "content": "hi"}]) == {"role": "assistant", "content": "hi"}


async def test_chat_posts_to_v1_chat_completions(mocker):
    seen = mock_http(mocker, json_reply(CHAT_OK))
    await chat("m", [], base_url="http://host:1234/")   # trailing slash must not double up
    assert str(seen[0].url) == "http://host:1234/v1/chat/completions"
    assert seen[0].method == "POST"


async def test_chat_payload_includes_model_and_disables_streaming(mocker):
    seen = mock_http(mocker, json_reply(CHAT_OK))
    await chat("my-model", [{"role": "user", "content": "q"}])
    body = json.loads(seen[0].content)
    assert body["model"] == "my-model" and body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "q"}]
    assert "tools" not in body and "response_format" not in body


async def test_chat_includes_tools_and_json_format_when_asked(mocker):
    seen = mock_http(mocker, json_reply(CHAT_OK))
    schemas = [{"type": "function", "function": {"name": "f"}}]
    await chat("m", [], tools=schemas, format="json")
    body = json.loads(seen[0].content)
    assert body["tools"] == schemas
    assert body["response_format"] == {"type": "json_object"}


async def test_chat_sends_bearer_header_when_key_present(mocker):
    seen = mock_http(mocker, json_reply(CHAT_OK))
    await chat("m", [], api_key="sk-abc")
    assert seen[0].headers["authorization"] == "Bearer sk-abc"


async def test_chat_omits_auth_header_when_no_key(mocker):
    """An unauthenticated local server must not get an empty Bearer header."""
    mocker.patch.object(llm_client, "DEFAULT_API_KEY", "")
    seen = mock_http(mocker, json_reply(CHAT_OK))
    await chat("m", [], api_key="")
    assert "authorization" not in seen[0].headers


async def test_chat_falls_back_to_env_key(mocker):
    mocker.patch.object(llm_client, "DEFAULT_API_KEY", "sk-from-env")
    seen = mock_http(mocker, json_reply(CHAT_OK))
    await chat("m", [])
    assert seen[0].headers["authorization"] == "Bearer sk-from-env"


async def test_chat_falls_back_to_default_base_url(mocker):
    mocker.patch.object(llm_client, "DEFAULT_BASE_URL", "http://fallback:9999")
    seen = mock_http(mocker, json_reply(CHAT_OK))
    await chat("m", [])
    assert str(seen[0].url).startswith("http://fallback:9999/")


async def test_chat_normalizes_dict_tool_call_arguments_on_the_wire(mocker):
    seen = mock_http(mocker, json_reply(CHAT_OK))
    await chat("m", [{"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}]}])
    sent = json.loads(seen[0].content)["messages"][0]
    assert sent["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


async def test_chat_passes_timeout_through(mocker):
    captured = {}
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        kwargs["transport"] = httpx.MockTransport(json_reply(CHAT_OK))
        return real(*args, **kwargs)

    mocker.patch.object(llm_client.httpx, "AsyncClient", factory)
    await chat("m", [], timeout=12.5)
    assert captured["timeout"] == 12.5


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
async def test_chat_raises_llmerror_on_http_error(mocker, status):
    mock_http(mocker, lambda r: httpx.Response(status, text="upstream boom"))
    with pytest.raises(LLMError, match=str(status)):
        await chat("m", [])


async def test_chat_raises_llmerror_on_connection_failure(mocker):
    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    mock_http(mocker, boom)
    with pytest.raises(LLMError, match="Could not reach"):
        await chat("m", [])


async def test_chat_raises_llmerror_on_timeout(mocker):
    def boom(request):
        raise httpx.ReadTimeout("too slow", request=request)

    mock_http(mocker, boom)
    with pytest.raises(LLMError):
        await chat("m", [])


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{}]}])
async def test_chat_raises_llmerror_on_unexpected_shape(mocker, payload):
    mock_http(mocker, json_reply(payload))
    with pytest.raises(LLMError, match="Unexpected response shape"):
        await chat("m", [])


# ---------------- list_models ----------------

async def test_list_models_returns_every_id_in_server_order(mocker):
    """Returned verbatim — the server's order is preserved and embedding
    models are NOT filtered out (the /model picker shows whatever the
    server reports)."""
    mock_http(mocker, json_reply({"data": [
        {"id": "qwen3.6:35b"},
        {"id": "llama3.1:latest"},
        {"id": "nomic-embed-text"},
    ]}))
    assert await list_models() == ["qwen3.6:35b", "llama3.1:latest", "nomic-embed-text"]


async def test_list_models_gets_v1_models_with_auth(mocker):
    seen = mock_http(mocker, json_reply({"data": []}))
    await list_models(base_url="http://h:1", api_key="sk-x")
    assert seen[0].method == "GET"
    assert str(seen[0].url) == "http://h:1/v1/models"
    assert seen[0].headers["authorization"] == "Bearer sk-x"


@pytest.mark.parametrize("items", [[{"no_id": 1}], ["junk"], [{"id": "ok"}, {"no_id": 2}]])
async def test_list_models_raises_on_entry_without_id(mocker, items):
    """A malformed entry is a protocol violation, so it surfaces as LLMError
    rather than being silently skipped."""
    mock_http(mocker, json_reply({"data": items}))
    with pytest.raises(LLMError, match="no 'id' key"):
        await list_models()


async def test_list_models_empty_list_is_not_an_error(mocker):
    mock_http(mocker, json_reply({"data": []}))
    assert await list_models() == []


async def test_list_models_raises_on_missing_data_key(mocker):
    mock_http(mocker, json_reply({"wrong": []}))
    with pytest.raises(LLMError, match="Unexpected response shape"):
        await list_models()


async def test_list_models_raises_on_http_error(mocker):
    mock_http(mocker, lambda r: httpx.Response(404, text="nope"))
    with pytest.raises(LLMError, match="404"):
        await list_models()


async def test_list_models_raises_on_connect_error(mocker):
    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    mock_http(mocker, boom)
    with pytest.raises(LLMError, match="Could not reach"):
        await list_models()


# ---------------- embed ----------------

async def test_embed_returns_vectors_in_order(mocker):
    mock_http(mocker, json_reply({"data": [
        {"embedding": [0.1, 0.2]},
        {"embedding": [0.3, 0.4]},
    ]}))
    assert await embed("m", ["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]


async def test_embed_posts_model_and_input(mocker):
    seen = mock_http(mocker, json_reply({"data": [{"embedding": [1.0]}]}))
    await embed("embed-model", ["text"], base_url="http://h:2")
    assert str(seen[0].url) == "http://h:2/v1/embeddings"
    body = json.loads(seen[0].content)
    assert body == {"model": "embed-model", "input": ["text"]}


@pytest.mark.parametrize("payload", [{}, {"data": []}, {"data": [{"nope": 1}]}])
async def test_embed_raises_on_bad_shape(mocker, payload):
    mock_http(mocker, json_reply(payload))
    with pytest.raises(LLMError):
        await embed("m", ["a"])


async def test_embed_raises_on_http_error(mocker):
    mock_http(mocker, lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError, match="500"):
        await embed("m", ["a"])


async def test_embed_raises_on_connect_error(mocker):
    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    mock_http(mocker, boom)
    with pytest.raises(LLMError, match="Could not reach"):
        await embed("m", ["a"])
