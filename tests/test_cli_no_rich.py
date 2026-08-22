"""The documented degraded mode: rich / prompt_toolkit not installed.

Every renderer in cli.py wraps `from . import ui` in try/except ImportError
and falls back to plain typer.echo, so the CLI still works on a bare
install. These force that branch by making the `omni.ui` import fail.
"""

import builtins

import pytest

from omni import cli as cli_mod


@pytest.fixture
def no_ui(mocker):
    """Make `from . import ui` raise, as it would without rich installed."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "omni.ui" or (fromlist and "ui" in fromlist and name in ("omni", "")):
            raise ImportError("no rich installed")
        return real_import(name, globals, locals, fromlist, level)

    mocker.patch.object(builtins, "__import__", side_effect=fake_import)


def test_header_falls_back_to_a_plain_line(no_ui, cfg, capsys):
    cli_mod._print_header(cfg, "my-session")
    out = capsys.readouterr().out
    assert cfg.model in out and "my-session" in out


def test_sessions_fall_back_to_plain_rows(no_ui, capsys):
    cli_mod._print_sessions([
        {"id": "abc12345", "name": "nm", "status": "done",
         "updated_at": "2026-01-01", "model": "m", "task": "the task"}])
    out = capsys.readouterr().out
    assert "abc12345" in out and "done" in out and "the task" in out


def test_sessions_plain_rows_handle_a_missing_name(no_ui, capsys):
    cli_mod._print_sessions([
        {"id": "i", "name": None, "status": "running",
         "updated_at": "t", "model": "m", "task": "t"}])
    assert "-" in capsys.readouterr().out


def test_resources_fall_back_to_plain_lines(no_ui, capsys):
    cli_mod._print_resources({
        "file:///a.md": {"server": "docs", "mime_type": "text/markdown",
                         "description": "Standards", "template": False},
        "file:///{d}.log": {"server": "docs", "mime_type": "", "description": "Daily",
                            "template": True},
    })
    out = capsys.readouterr().out
    assert "file:///a.md" in out and "text/markdown" in out and "Standards" in out
    assert "template" in out
    assert "/resources <uri>" in out


def test_empty_resources_fall_back_to_a_notice(no_ui, capsys):
    cli_mod._print_resources({})
    assert "No resources" in capsys.readouterr().out


def test_mcp_status_falls_back_to_plain_lines(no_ui, capsys):
    cli_mod._print_mcp_status([
        {"name": "built-in", "connected": True, "connected_for": 65.0, "error": None,
         "deferred": False, "tool_count": 18, "target": "python -m omni.mcp_server"},
        {"name": "docs", "connected": False, "connected_for": None, "error": "boom",
         "deferred": False, "tool_count": 0, "target": "https://h/mcp"},
    ])
    captured = capsys.readouterr()
    assert "[OK]" in captured.out and "built-in" in captured.out
    assert "1m 05s" in captured.out and "18 tools" in captured.out
    assert "[FAIL]" in captured.err and "boom" in captured.err


def test_resumed_history_falls_back_to_plain_lines(no_ui, mocker, tmp_path, capsys):
    from omni.session_store import SessionStore
    store = SessionStore(str(tmp_path / "s.db"))
    sid = store.create_session("/p", "m", "t")
    store.append_message(sid, 0, {"role": "system", "content": "the system prompt"})
    store.append_message(sid, 1, {"role": "user", "content": "my question"})

    cli_mod._show_resumed_history(str(tmp_path / "s.db"), sid)
    out = capsys.readouterr().out
    assert "Resumed history" in out and "my question" in out
    assert "the system prompt" not in out       # system messages stay hidden


async def test_btw_falls_back_to_plain_echo(no_ui, cfg, mocker, capsys):
    mocker.patch.object(cli_mod, "chat", mocker.AsyncMock(return_value={"content": "the answer"}))
    await cli_mod._handle_btw(cfg, "the question")
    out = capsys.readouterr().out
    assert "the question" in out and "the answer" in out


async def test_restart_falls_back_to_no_spinner(no_ui, mocker):
    client = mocker.AsyncMock()
    client.restart_server.return_value = {"name": "docs", "connected": True, "tool_count": 1}
    out = await cli_mod._restart_mcp_server(client, "docs")
    assert out["connected"] is True            # works without the spinner


async def test_read_task_falls_back_to_input(mocker):
    """prompt_session is None when prompt_toolkit is missing."""
    mocker.patch("builtins.input", return_value="typed plainly")
    assert await cli_mod._read_task(None) == "typed plainly"


def test_server_tools_fall_back_to_plain_lines(no_ui, capsys):
    cli_mod._print_server_tools("docs", [
        {"name": "docs__search", "real_name": "search", "description": "Search the docs",
         "deferred": True, "revealed": False, "internal": False},
        {"name": "docs__publish", "real_name": "publish", "description": "Publish",
         "deferred": False, "revealed": False, "internal": False},
    ])
    out = capsys.readouterr().out
    assert "docs__search" in out and "Search the docs" in out
    assert "[deferred]" in out
    assert "docs__publish" in out


def test_empty_server_tools_fall_back_to_a_notice(no_ui, capsys):
    cli_mod._print_server_tools("docs", [])
    assert "no tools" in capsys.readouterr().out
