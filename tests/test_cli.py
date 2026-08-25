"""cli.py — the Typer surface (flags, one-shot runs, MCP registry commands,
session management) and the REPL's helper functions.

CodingAgent.run and the MCP client are mocked, so nothing spawns a
subprocess or reaches a model; HOME is redirected so the real
~/.omni-coder settings file is never touched.
"""

import json

import pytest
from typer.testing import CliRunner

from omni import cli as cli_mod
from omni.cli import app
from omni.config import AgentConfig
from omni.llm_client import LLMError

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_home(mocker, tmp_path):
    """Point ~ at tmp_path so the global settings file stays untouched."""
    home = tmp_path / "home"
    (home / ".omni-coder").mkdir(parents=True)
    mocker.patch("os.path.expanduser", lambda p: p.replace("~", str(home)))
    return home


@pytest.fixture
def settings_path(isolated_home):
    return isolated_home / ".omni-coder" / "omni-coder-settings.json"


@pytest.fixture
def no_interactive(mocker):
    """Guard: a bare `omni` invocation must never open a real REPL in tests."""
    return mocker.patch.object(cli_mod, "_interactive", mocker.AsyncMock())


def invoke(*args, **kwargs):
    return runner.invoke(app, list(args), **kwargs)


# ---------------- one-shot run ----------------

def test_one_shot_run_prints_the_result(mocker, tmp_path):
    run = mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="THE RESULT"))
    r = invoke("do the thing", "--db-path", str(tmp_path / "s.db"),
               "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 0 and "THE RESULT" in r.output
    assert run.await_args.args[0] == "do the thing"


def test_flags_are_threaded_into_agentconfig(mocker, tmp_path):
    captured = {}
    mocker.patch.object(cli_mod.CodingAgent, "__init__",
                        lambda self, cfg: captured.update(cfg=cfg))
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="ok"))
    invoke("t", "--project-root", str(tmp_path), "--model", "my-model",
           "--llm-host", "http://h:1", "--llm-api-key", "sk-k", "--llm-timeout", "42",
           "--max-steps", "7", "--auto-approve", "--skip-intent-parsing",
           "--intent-model", "small", "--compact-model", "tiny",
           "--compact-keep-last", "9", "--context-char-budget", "1234",
           "--embedding-model", "", "--db-path", str(tmp_path / "d.db"),
           "--log-path", str(tmp_path / "l.log"))
    cfg = captured["cfg"]
    assert cfg.model == "my-model" and cfg.llm_host == "http://h:1"
    assert cfg.llm_api_key == "sk-k" and cfg.llm_timeout_s == 42
    assert cfg.max_steps == 7 and cfg.auto_approve is True
    assert cfg.parse_intent is False and cfg.intent_model == "small"
    assert cfg.compact_model == "tiny" and cfg.compact_keep_last == 9
    assert cfg.context_char_budget == 1234 and cfg.embedding_model == ""


def test_embedding_model_defaults_when_flag_omitted(mocker, tmp_path):
    captured = {}
    mocker.patch.object(cli_mod.CodingAgent, "__init__", lambda self, cfg: captured.update(cfg=cfg))
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="ok"))
    invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].embedding_model == AgentConfig.embedding_model


def test_safe_tool_flag_extends_the_auto_approved_set(mocker, tmp_path):
    captured = {}
    mocker.patch.object(cli_mod.CodingAgent, "__init__", lambda self, cfg: captured.update(cfg=cfg))
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="ok"))
    invoke("t", "--safe-tool", "docs__search", "--safe-tool", "docs__lookup",
           "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    safe = captured["cfg"].safe_tools
    assert "docs__search" in safe and "docs__lookup" in safe
    assert "read_file" in safe          # the built-in read-only set is kept


def test_defer_applies_to_inline_mcp_servers(mocker, tmp_path):
    captured = {}
    mocker.patch.object(cli_mod.CodingAgent, "__init__", lambda self, cfg: captured.update(cfg=cfg))
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="ok"))
    invoke("t", "--mcp-server", "docs=node srv.js", "--defer",
           "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].mcp_servers["docs"]["defer"] is True


def test_defer_without_any_server_says_so(mocker, tmp_path):
    """It used to be silently ignored unless paired with --add-mcp-server."""
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="ok"))
    r = invoke("t", "--defer", "--db-path", str(tmp_path / "d.db"),
               "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 0 and "--defer had no effect" in r.output


def test_run_value_error_exits_nonzero(mocker, tmp_path):
    mocker.patch.object(cli_mod.CodingAgent, "run",
                        mocker.AsyncMock(side_effect=ValueError("no session found")))
    r = invoke("t", "--db-path", str(tmp_path / "s.db"), "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1 and "no session found" in r.output


def test_unreachable_llm_server_is_one_line_not_a_traceback(mocker, tmp_path):
    """What _call_model raises once its retries are spent — the single most
    common failure mode, and previously an uncaught stack trace."""
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(
        side_effect=RuntimeError("Model call failed after 3 attempts: Could not reach the LLM server")))
    r = invoke("t", "--db-path", str(tmp_path / "s.db"), "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1
    assert "Could not reach the LLM server" in r.output
    assert "Traceback" not in r.output


def test_llm_error_from_a_run_exits_nonzero(mocker, tmp_path):
    mocker.patch.object(cli_mod.CodingAgent, "run",
                        mocker.AsyncMock(side_effect=LLMError("bad gateway")))
    r = invoke("t", "--db-path", str(tmp_path / "s.db"), "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1 and "bad gateway" in r.output


def test_no_task_enters_interactive(no_interactive, tmp_path):
    r = invoke("--db-path", str(tmp_path / "s.db"), "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 0
    no_interactive.assert_awaited_once()


def test_help_lists_the_key_flags():
    out = invoke("--help").output
    for flag in ("--project-root", "--model", "--llm-host", "--llm-timeout",
                 "--auto-approve", "--safe-tool", "--resume", "--add-mcp-server",
                 "--mcp-log-path"):
        assert flag in out


# ---------------- MCP registry commands ----------------

def test_add_mcp_server_writes_to_the_settings_file(settings_path):
    r = invoke("--add-mcp-server", "weather=python -m weather_srv")
    assert r.exit_code == 0 and "Registered" in r.output
    data = json.loads(settings_path.read_text())
    assert data["mcpServers"]["weather"]["command"] == "python"
    assert data["mcpServers"]["weather"]["args"] == ["-m", "weather_srv"]


def test_add_mcp_server_with_defer_flag(settings_path):
    invoke("--add-mcp-server", "docs=node srv.js", "--defer")
    assert json.loads(settings_path.read_text())["mcpServers"]["docs"]["defer"] is True


def test_add_mcp_server_with_bearer_keeps_env_reference(settings_path):
    invoke("--add-mcp-server", "docs=https://h/mcp/sse,bearer=$DOCS_TOKEN")
    entry = json.loads(settings_path.read_text())["mcpServers"]["docs"]
    assert entry["headers"] == {"Authorization": "Bearer $DOCS_TOKEN"}


def test_add_mcp_server_preserves_other_settings_keys(settings_path):
    settings_path.write_text(json.dumps({"theme": "dark", "mcpServers": {}}))
    invoke("--add-mcp-server", "w=python -m w")
    data = json.loads(settings_path.read_text())
    assert data["theme"] == "dark" and "w" in data["mcpServers"]


def test_add_mcp_server_rejects_malformed_spec(settings_path):
    r = invoke("--add-mcp-server", "no-equals-sign")
    assert r.exit_code == 1 and "Error" in r.output
    assert not settings_path.exists()


def test_list_mcp_servers(settings_path):
    settings_path.write_text(json.dumps({"mcpServers": {
        "w": {"command": "python", "args": ["-m", "w"]},
        "d": {"url": "https://h/mcp", "defer": True},
    }}))
    out = invoke("--list-mcp-servers").output
    assert "w: python -m w" in out
    assert "d: https://h/mcp [defer]" in out


def test_list_mcp_servers_when_empty():
    assert "No registered MCP servers" in invoke("--list-mcp-servers").output


def test_remove_mcp_server(settings_path):
    settings_path.write_text(json.dumps({"mcpServers": {"w": {"command": "python"}}}))
    r = invoke("--remove-mcp-server", "w")
    assert r.exit_code == 0 and "Removed" in r.output
    assert json.loads(settings_path.read_text())["mcpServers"] == {}


def test_remove_unknown_mcp_server_errors(settings_path):
    settings_path.write_text(json.dumps({"mcpServers": {}}))
    r = invoke("--remove-mcp-server", "ghost")
    assert r.exit_code == 1 and "no registered MCP server" in r.output


# ---------------- session management commands ----------------

@pytest.fixture
def store(tmp_path):
    from omni.session_store import SessionStore
    return SessionStore(str(tmp_path / "s.db"))


def test_list_sessions(store, tmp_path):
    store.create_session("/p", "qwen", "my saved task", name="nm")
    out = invoke("--list-sessions", "--db-path", str(tmp_path / "s.db")).output
    assert "my saved task" in out and "nm" in out


def test_list_sessions_when_empty(tmp_path):
    out = invoke("--list-sessions", "--db-path", str(tmp_path / "empty.db")).output
    assert "No saved sessions" in out


def test_delete_session(store, tmp_path):
    sid = store.create_session("/p", "m", "t", name="doomed")
    r = invoke("--delete-session", "doomed", "--db-path", str(tmp_path / "s.db"))
    assert r.exit_code == 0 and "Deleted" in r.output
    assert not store.session_exists(sid)


def test_delete_unknown_session_errors(tmp_path):
    r = invoke("--delete-session", "ghost", "--db-path", str(tmp_path / "s.db"))
    assert r.exit_code == 1 and "no session found" in r.output.lower()


# ---------------- _handle_btw ----------------

async def test_handle_btw_answers_without_touching_history(cfg, mocker):
    chat = mocker.patch.object(cli_mod, "chat", mocker.AsyncMock(return_value={"content": "42"}))
    shown = mocker.patch("omni.ui.btw_answer")
    await cli_mod._handle_btw(cfg, "what is 6*7?")

    shown.assert_called_once_with("what is 6*7?", "42")
    sent = chat.await_args.kwargs["messages"]
    assert len(sent) == 2 and sent[1]["content"] == "what is 6*7?"   # stateless, no history


async def test_handle_btw_uses_configured_model_and_host(cfg, mocker):
    cfg.llm_host, cfg.llm_api_key, cfg.llm_timeout_s = "http://h", "k", 33.0
    chat = mocker.patch.object(cli_mod, "chat", mocker.AsyncMock(return_value={"content": "x"}))
    mocker.patch("omni.ui.btw_answer")
    await cli_mod._handle_btw(cfg, "q")
    kwargs = chat.await_args.kwargs
    assert kwargs["model"] == cfg.model and kwargs["base_url"] == "http://h"
    assert kwargs["api_key"] == "k" and kwargs["timeout"] == 33.0


async def test_handle_btw_reports_llm_errors_inline(cfg, mocker):
    mocker.patch.object(cli_mod, "chat", mocker.AsyncMock(side_effect=LLMError("server down")))
    shown = mocker.patch("omni.ui.btw_answer")
    await cli_mod._handle_btw(cfg, "q")
    assert "server down" in shown.call_args.args[1]


async def test_handle_btw_handles_empty_response(cfg, mocker):
    mocker.patch.object(cli_mod, "chat", mocker.AsyncMock(return_value={"content": None}))
    shown = mocker.patch("omni.ui.btw_answer")
    await cli_mod._handle_btw(cfg, "q")
    assert "(empty response)" in shown.call_args.args[1]


# ---------------- REPL helpers ----------------

async def test_read_task_uses_prompt_session_when_available(mocker):
    ui_read = mocker.patch("omni.ui.prompt_task_async", mocker.AsyncMock(return_value="typed"))
    assert await cli_mod._read_task(mocker.Mock()) == "typed"
    ui_read.assert_awaited_once()


async def test_read_task_falls_back_to_input(mocker):
    mocker.patch("builtins.input", return_value="typed at plain prompt")
    assert await cli_mod._read_task(None) == "typed at plain prompt"


def test_print_header_delegates_to_ui(mocker, cfg):
    """No model in the header: it would be stale after the first /model
    switch, and the frame's hint line carries the live one instead."""
    header = mocker.patch("omni.ui.header")
    cli_mod._print_header(cfg, "my-label")
    header.assert_called_once_with("my-label", cfg.project_root)


def test_model_switch_is_announced_without_a_header(mocker):
    switched = mocker.patch("omni.ui.model_switched")
    header = mocker.patch("omni.ui.header")
    cli_mod._announce_model("llama3.1:latest")
    switched.assert_called_once_with("llama3.1:latest")
    header.assert_not_called()


def test_print_sessions_empty_and_populated(mocker, capsys):
    table = mocker.patch("omni.ui.sessions_table")
    cli_mod._print_sessions([])
    assert "No saved sessions" in capsys.readouterr().out
    table.assert_not_called()

    rows = [{"id": "a", "name": None, "status": "done", "updated_at": "t", "model": "m", "task": "x"}]
    cli_mod._print_sessions(rows)
    table.assert_called_once_with(rows)


def test_print_resources_delegates_to_ui(mocker):
    table = mocker.patch("omni.ui.resources_table")
    cli_mod._print_resources({"x://1": {"server": "d", "template": False}})
    table.assert_called_once()


def test_print_server_tools_delegates_to_ui(mocker):
    table = mocker.patch("omni.ui.server_tools_table")
    tools = [{"name": "docs__search", "real_name": "search", "description": "d",
              "deferred": False, "revealed": False, "internal": False}]
    cli_mod._print_server_tools("docs", tools)
    table.assert_called_once_with("docs", tools)


def test_print_mcp_status_delegates_to_ui(mocker):
    status = mocker.patch("omni.ui.mcp_status")
    cli_mod._print_mcp_status([{"name": "built-in", "connected": True}])
    status.assert_called_once()


def test_show_resumed_history_renders_when_session_exists(mocker, tmp_path):
    from omni.session_store import SessionStore
    store = SessionStore(str(tmp_path / "s.db"))
    sid = store.create_session("/p", "m", "t")
    store.append_message(sid, 0, {"role": "user", "content": "earlier"})
    panel = mocker.patch("omni.ui.history_panel")
    cli_mod._show_resumed_history(str(tmp_path / "s.db"), sid)
    assert panel.call_args.args[0][0]["content"] == "earlier"


def test_show_resumed_history_silent_for_unknown_session(mocker, tmp_path):
    """Stays quiet so agent.run() can raise the proper error instead."""
    panel = mocker.patch("omni.ui.history_panel")
    cli_mod._show_resumed_history(str(tmp_path / "s.db"), "ghost")
    panel.assert_not_called()


async def test_restart_mcp_server_wraps_the_client_call(mocker):
    client = mocker.AsyncMock()
    client.restart_server.return_value = {"name": "docs", "connected": True, "tool_count": 3}
    mocker.patch("omni.ui.thinking")
    out = await cli_mod._restart_mcp_server(client, "docs")
    assert out["connected"] is True
    client.restart_server.assert_awaited_once_with("docs")


# ---------------- static command registry ----------------

@pytest.mark.parametrize("command", [
    "/exit", "/quit", "/sessions", "/delete ", "/compact",
    "/btw ", "/model", "/mcp", "/mcp restart ", "/resources",
])
def test_static_commands_are_registered_with_descriptions(command):
    assert command in cli_mod._STATIC_COMMANDS
    assert cli_mod._STATIC_COMMANDS[command].strip()
