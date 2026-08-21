"""Embedding backends (search_tools' semantic ranking) and the
`python -m omni` entry point.

The `nomic` package is an optional extra that may not be installed, so it's
faked in sys.modules — that also lets the "not installed" branch be tested
on a machine where it *is* installed, and vice versa.
"""

import runpy
import subprocess
import sys

import pytest

from omni import mcp_client as mc
from omni.mcp_client import EmbeddingUnavailable, MCPToolClient, _LOCAL_NOMIC_MODEL
from omni.llm_client import LLMError


@pytest.fixture
def client(tmp_path):
    return MCPToolClient(str(tmp_path), mcp_log_path=str(tmp_path / "mcp.log"))


# ---------------- _embed_texts routing ----------------

async def test_embed_texts_routes_to_local_nomic_by_default(client, mocker):
    client.embedding_model = _LOCAL_NOMIC_MODEL
    local = mocker.patch.object(client, "_embed_local_nomic", mocker.AsyncMock(return_value=[[1.0]]))
    assert await client._embed_texts(["x"], "search_query") == [[1.0]]
    local.assert_awaited_once_with(["x"], "search_query")


async def test_embed_texts_routes_to_the_remote_endpoint(client, mocker):
    client.embedding_model = "mxbai-embed-large"
    client.llm_host, client.llm_api_key = "http://h", "k"
    remote = mocker.patch.object(mc, "embed", mocker.AsyncMock(return_value=[[0.5]]))
    assert await client._embed_texts(["x"], "search_document") == [[0.5]]
    assert remote.await_args.args[0] == "mxbai-embed-large"
    assert remote.await_args.kwargs["base_url"] == "http://h"
    assert remote.await_args.kwargs["api_key"] == "k"


async def test_embed_texts_converts_llm_errors_to_embedding_unavailable(client, mocker):
    """search_mcp_tools catches EmbeddingUnavailable to fall back to keyword
    matching — a raw LLMError here would break the tool call instead."""
    client.embedding_model = "remote-model"
    mocker.patch.object(mc, "embed", mocker.AsyncMock(side_effect=LLMError("502 bad gateway")))
    with pytest.raises(EmbeddingUnavailable, match="502"):
        await client._embed_texts(["x"], "search_query")


# ---------------- _embed_local_nomic ----------------

def fake_nomic(mocker, embeddings=None, error=None):
    """Install a stand-in `nomic` package exposing embed.text()."""
    embed_mod = mocker.Mock()
    if error is not None:
        embed_mod.text.side_effect = error
    else:
        embed_mod.text.return_value = {"embeddings": embeddings}
    pkg = mocker.Mock(embed=embed_mod)
    mocker.patch.dict(sys.modules, {"nomic": pkg})
    return embed_mod


async def test_local_nomic_returns_embeddings(client, mocker):
    embed_mod = fake_nomic(mocker, embeddings=[[0.1, 0.2]])
    out = await client._embed_local_nomic(["some text"], "search_document")
    assert out == [[0.1, 0.2]]
    kwargs = embed_mod.text.call_args.kwargs
    assert kwargs["texts"] == ["some text"]
    assert kwargs["task_type"] == "search_document"
    assert kwargs["inference_mode"] == "local"       # never phones home
    assert kwargs["model"] == mc._NOMIC_MODEL_NAME


async def test_local_nomic_missing_package_is_actionable(client, mocker):
    """The optional extra isn't installed — the message must say how to fix
    it, and be catchable as EmbeddingUnavailable."""
    mocker.patch.dict(sys.modules, {"nomic": None})
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fail_nomic(name, *args, **kwargs):
        if name == "nomic":
            raise ImportError("no nomic")
        return real_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=fail_nomic)
    with pytest.raises(EmbeddingUnavailable, match="nomic\\[local\\]"):
        await client._embed_local_nomic(["x"], "search_query")


async def test_local_nomic_runtime_failure_is_wrapped(client, mocker):
    fake_nomic(mocker, error=RuntimeError("model download failed"))
    with pytest.raises(EmbeddingUnavailable, match="model download failed"):
        await client._embed_local_nomic(["x"], "search_query")


async def test_local_nomic_runs_off_the_event_loop(client, mocker):
    """nomic.embed.text blocks, so it must go through a thread rather than
    stalling the loop."""
    fake_nomic(mocker, embeddings=[[1.0]])
    to_thread = mocker.patch.object(mc.asyncio, "to_thread", mocker.AsyncMock(return_value=[[1.0]]))
    await client._embed_local_nomic(["x"], "search_query")
    to_thread.assert_awaited_once()


async def test_semantic_search_falls_back_when_nomic_is_missing(client, mocker):
    """End to end: the default on-device backend being unavailable degrades
    search_tools to keyword matching instead of failing the call."""
    def tool(name, description):
        t = type("T", (), {})()
        t.name, t.description, t.inputSchema = name, description, {"type": "object", "properties": {}}
        return t

    session = mocker.AsyncMock()
    session.list_tools.return_value = mocker.Mock(tools=[tool("forecast", "weather forecast")])
    client._sessions["toy"] = session
    client._deferred_servers.add("toy")
    client.embedding_model = _LOCAL_NOMIC_MODEL
    await client.list_llm_tools()

    mocker.patch.object(client, "_embed_local_nomic",
                        mocker.AsyncMock(side_effect=EmbeddingUnavailable("not installed")))
    out = await client.search_mcp_tools("forecast")
    assert "fell back to keyword match" in out and "toy__forecast" in out


# ---------------- python -m omni ----------------

def test_module_entrypoint_is_wired_to_the_cli(mocker):
    app = mocker.patch("omni.cli.app")
    runpy.run_module("omni", run_name="__main__")
    app.assert_called_once()


def test_module_entrypoint_runs_as_a_subprocess():
    """`python -m omni --help` must work for real, not just when imported."""
    r = subprocess.run([sys.executable, "-m", "omni", "--help"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    assert "--project-root" in r.stdout


def test_console_script_entrypoint_is_declared():
    import tomllib
    from pathlib import Path
    data = tomllib.loads(Path("pyproject.toml").read_text())
    assert data["project"]["scripts"]["omni"] == "omni.cli:app"
