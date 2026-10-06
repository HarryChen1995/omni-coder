"""The interactive REPL loop (_interactive).

Driven by scripting `_read_task` to return a sequence of typed lines ending
in /exit, with the MCP client and CodingAgent.run mocked. That exercises
every slash command's branch plus the turn plumbing — session id carry-over,
header refresh, cancellation — without a terminal or a model.
"""

import asyncio
import contextlib
import json

import pytest

from omni import cli as cli_mod


@pytest.fixture(autouse=True)
def isolated_home(mocker, tmp_path):
    """Point ~ at tmp_path so the real ~/.omni-coder settings file is never
    touched by a /mcp remove in these tests."""
    home = tmp_path / "home"
    (home / ".omni-coder").mkdir(parents=True)
    mocker.patch("os.path.expanduser", lambda p: p.replace("~", str(home)))
    return home


@pytest.fixture
def settings_path(isolated_home):
    return isolated_home / ".omni-coder" / "omni-coder-settings.json"


@pytest.fixture
def client(mocker):
    """An MCPToolClient stand-in, installed as an async context manager."""
    c = mocker.AsyncMock()
    c.list_llm_tools.return_value = []
    c.list_prompts.return_value = {}
    c.list_resources.return_value = {}
    # server_status/server_names are SYNC methods — an AsyncMock would hand
    # back coroutines and the REPL iterates the result directly.
    c.server_status = mocker.Mock(return_value=[
        {"name": "built-in", "connected": True, "connected_for": 1.0, "error": None,
         "deferred": False, "tool_count": 18, "target": "python -m omni.mcp_server"}])
    c.server_names = mocker.Mock(return_value=["built-in"])
    c.__aenter__.return_value = c
    c.__aexit__.return_value = False
    mocker.patch.object(cli_mod, "MCPToolClient", return_value=c)
    return c


def scripted(mocker, lines):
    """Feed the REPL a sequence of typed lines.

    Input now arrives as (pane, text) — the pane says which agent it was
    typed at — so the stand-in hands back whichever pane the loop offered it,
    which is main."""
    remaining = iter(lines)

    async def fake_next(tui, main_pane):
        # Turns are dispatched rather than awaited, so wait for the pane to be
        # free before "typing" the next line — which is what a person does,
        # and what makes these sequences deterministic.
        for _ in range(200):
            if main_pane.task is None or main_pane.task.done():
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0)      # let the full-screen app, if any, start
        try:
            return main_pane, next(remaining)
        except StopIteration:
            return None, None

    return mocker.patch.object(cli_mod, "_next_instruction", fake_next)


@pytest.fixture
def repl(mocker, client, cfg):
    """Run the REPL over a scripted list of inputs. Returns a helper that
    yields the mocks for assertions."""
    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch("omni.ui.final_result")
    mocker.patch("omni.ui.interrupted")
    mocker.patch("omni.ui.thinking")

    captured = {}

    def run(inputs, run_result="the answer", run_side_effect=None, **kwargs):
        script = list(inputs) + ["/exit"]

        # No full-screen app in these tests: they assert on what reaches the
        # terminal, and the app would swallow it into its own transcript. The
        # TUI has its own tests (test_tui.py) plus the pty checks. The
        # commands dict it would have been given is captured for the
        # completion tests, which is where that dict is assembled.
        def fake_make_tui(commands, session_label, model):
            captured["commands"] = commands
            captured["label"] = session_label      # what the frame's chip says
            return None

        mocker.patch.object(cli_mod, "_make_tui", fake_make_tui)
        scripted(mocker, script)
        agent_run = mocker.patch.object(
            cli_mod.CodingAgent, "run",
            mocker.AsyncMock(return_value=run_result, side_effect=run_side_effect))
        asyncio.run(cli_mod._interactive(cfg, kwargs.get("resume"), kwargs.get("session_name")))
        return agent_run

    run.captured = captured
    return run


# ---------------- turn plumbing ----------------

def test_a_typed_task_runs_a_turn(repl, mocker):
    result = mocker.patch("omni.ui.final_result")
    agent_run = repl(["do the thing"])
    assert agent_run.await_args.args[0] == "do the thing"
    result.assert_called_once_with("the answer")


def test_blank_input_is_ignored(repl):
    agent_run = repl(["", "   "])
    agent_run.assert_not_awaited()


@pytest.mark.parametrize("quit_cmd", ["/exit", "/quit"])
def test_exit_commands_leave_the_loop(repl, mocker, quit_cmd):
    """Anything typed after the quit command must never run."""
    mocker.patch.object(cli_mod, "_make_tui", lambda *a, **k: None)
    scripted(mocker, [quit_cmd, "never runs"])
    agent_run = mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock())
    repl([quit_cmd])
    agent_run.assert_not_awaited()


def test_session_id_is_carried_between_turns(mocker, client, cfg):
    """Turn 2 must resume the session turn 1 created, not start a fresh one."""
    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch("omni.ui.final_result")
    mocker.patch.object(cli_mod, "_make_tui", lambda *a, **k: None)
    scripted(mocker, ["first", "second", "/exit"])
    mocker.patch("prompt_toolkit.PromptSession", mocker.Mock())
    mocker.patch("prompt_toolkit.patch_stdout.patch_stdout", mocker.MagicMock())

    seen = []

    async def fake_run(self, task, **kwargs):
        seen.append(kwargs.get("resume_session_id"))
        self.session_id = "abc12345"          # what a real turn assigns
        return "ok"

    mocker.patch.object(cli_mod.CodingAgent, "run", fake_run)
    asyncio.run(cli_mod._interactive(cfg, None, None))
    assert seen == [None, "abc12345"]


def test_the_same_client_is_reused_across_turns(repl, client):
    repl(["one", "two"])
    assert cli_mod.MCPToolClient.call_count == 1     # one tool server for the session


def test_eof_at_the_prompt_exits_cleanly(mocker, client, cfg):
    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch.object(cli_mod, "_make_tui", lambda *a, **k: None)
    mocker.patch.object(cli_mod, "_next_instruction", mocker.AsyncMock(side_effect=EOFError))
    mocker.patch("prompt_toolkit.PromptSession", mocker.Mock())
    mocker.patch("prompt_toolkit.patch_stdout.patch_stdout", mocker.MagicMock())
    asyncio.run(cli_mod._interactive(cfg, None, None))   # must not raise


def test_turn_value_error_is_reported_and_the_loop_survives(repl, capsys):
    agent_run = repl(["bad task", "good task"], run_side_effect=[ValueError("nope"), "ok"])
    assert agent_run.await_count == 2                 # kept going after the error
    assert "nope" in capsys.readouterr().err


def test_turn_runtime_error_is_reported_and_the_loop_survives(repl, capsys):
    """_call_model gives up with a RuntimeError when the LLM server is
    unreachable; that used to unwind past the loop and end the REPL, taking
    the MCP connections with it."""
    agent_run = repl(["first", "second"],
                     run_side_effect=[RuntimeError("Model call failed after 3 attempts"), "ok"])
    assert agent_run.await_count == 2
    assert "Model call failed" in capsys.readouterr().err



def test_cancelled_turn_keeps_the_repl_alive(repl, mocker):
    interrupted = mocker.patch("omni.ui.interrupted")
    agent_run = repl(["long task", "next task"],
                     run_side_effect=[asyncio.CancelledError(), "done"])
    interrupted.assert_called_once()
    assert agent_run.await_count == 2


def test_the_repl_client_carries_the_tool_side_config(repl, cfg):
    """The built-in server is a subprocess; its shell timeout, truncation
    limit, memory file and denylist only reach it through this env."""
    repl([])
    env = cli_mod.MCPToolClient.call_args.kwargs["builtin_env"]
    assert env["AGENT_PROJECT_ROOT"] == cfg.project_root
    assert env["AGENT_SHELL_TIMEOUT_S"] == str(cfg.shell_timeout_s)
    assert env["AGENT_MEMORY_PATH"] == cfg.memory_path


def test_terminal_is_titled_after_the_session(repl, mocker, cfg):
    title = mocker.patch("omni.ui.set_terminal_title")
    repl([], session_name="white-house-3d")
    assert title.call_args_list[0].args[0] == "white-house-3d"


def test_title_follows_a_resumed_session(repl, mocker):
    title = mocker.patch("omni.ui.set_terminal_title")
    repl([], resume="utils-typing")
    assert title.call_args_list[0].args[0] == "utils-typing"


# ---------------- what a resumed session is called ----------------

def saved(cfg, name=None, task="earlier work"):
    """A session already in the DB the REPL is about to open."""
    from omni.session_store import SessionStore
    return SessionStore(cfg.db_path).create_session(cfg.project_root, cfg.model, task, name=name)


def test_a_resumed_session_is_labelled_with_its_name(repl, cfg):
    """Not "<hex id> (resumed)". The name is what was chosen and what was
    typed to get back here; that it was resumed is the least interesting
    thing about it once you are sitting in it."""
    saved(cfg, name="my refactor")
    repl([], resume="my refactor")
    assert repl.captured["label"] == "my refactor"


def test_resuming_by_id_still_shows_the_name(repl, cfg):
    """The label says what the session is, not what you typed to reach it."""
    sid = saved(cfg, name="my refactor")
    repl([], resume=sid)
    assert repl.captured["label"] == "my refactor"


def test_an_unnamed_session_falls_back_to_its_id(repl, cfg):
    """There is nothing else to call it."""
    sid = saved(cfg)
    repl([], resume=sid)
    assert repl.captured["label"] == sid


def test_the_resumed_label_carries_no_suffix(repl, cfg):
    sid = saved(cfg, name="my refactor")
    repl([], resume=sid)
    assert "(resumed)" not in repl.captured["label"]


def test_a_resume_naming_nothing_is_left_for_the_turn_to_complain_about(repl, cfg):
    """Unresolvable names must not be swallowed here — agent.run() raises the
    proper "no session found" on the first turn, and the label has only the
    raw value to show meanwhile."""
    repl([], resume="never-existed")
    assert repl.captured["label"] == "never-existed"


def test_a_new_named_session_is_labelled_with_its_name(repl):
    repl([], session_name="white-house-3d")
    assert repl.captured["label"] == "white-house-3d"


def test_a_fresh_unnamed_session_says_it_is_new(repl):
    repl([])
    assert repl.captured["label"] == "(new)"


def test_resuming_by_name_finds_the_session_whatever_the_case(repl, cfg):
    """The point of quoting a name: it is typed from memory."""
    sid = saved(cfg, name="My Refactor")
    run = repl(["carry on then"], resume="  my refactor  ")
    assert repl.captured["label"] == "My Refactor"
    # and the turn runs against that session, not a new one
    assert run.await_args.kwargs["resume_session_id"] == sid


def test_the_window_title_uses_the_name_too(repl, mocker, cfg):
    title = mocker.patch("omni.ui.set_terminal_title")
    sid = saved(cfg, name="my refactor")
    repl([], resume=sid)
    assert title.call_args_list[0].args[0] == "my refactor"


def test_unnamed_session_is_titled_once_it_has_an_id(repl, mocker):
    title = mocker.patch("omni.ui.set_terminal_title")
    real_init = cli_mod.CodingAgent.__init__

    def seeded_init(self, cfg):
        real_init(self, cfg)
        self.session_id = "abc12345"

    mocker.patch.object(cli_mod.CodingAgent, "__init__", seeded_init)
    repl(["a task"])
    assert "abc12345" in [c.args[0] for c in title.call_args_list]


def test_reasoning_command_expands_the_last_chain_of_thought(repl, mocker):
    full = mocker.patch("omni.ui.reasoning_full")
    # last_reasoning is set per instance in __init__, so patching the class
    # attribute would be shadowed — seed it as the REPL builds its agent.
    real_init = cli_mod.CodingAgent.__init__

    def seeded_init(self, cfg):
        real_init(self, cfg)
        self.last_reasoning = "the thinking"

    mocker.patch.object(cli_mod.CodingAgent, "__init__", seeded_init)
    repl(["/reasoning"])
    full.assert_called_once_with("the thinking")


def test_reasoning_command_without_any_says_so(repl, mocker, capsys):
    mocker.patch.object(cli_mod.CodingAgent, "last_reasoning", "", create=True)
    agent_run = repl(["/reasoning"])
    assert "No reasoning recorded" in capsys.readouterr().out
    agent_run.assert_not_awaited()


def test_mcp_remove_disconnects_and_unregisters(repl, client, mocker, settings_path, capsys):
    client.remove_server = mocker.AsyncMock(return_value="docs")
    settings_path.write_text(json.dumps({"mcpServers": {"docs": {"command": "node srv.js"}}}))
    repl(["/mcp remove docs"])
    client.remove_server.assert_awaited_once_with("docs")
    assert "docs" not in json.loads(settings_path.read_text())["mcpServers"]
    assert "unregistered" in capsys.readouterr().out


def test_mcp_remove_of_an_inline_server_only_affects_the_session(repl, client, mocker,
                                                                 settings_path, capsys):
    client.remove_server = mocker.AsyncMock(return_value="inline")
    settings_path.write_text(json.dumps({"mcpServers": {}}))
    repl(["/mcp remove inline"])
    assert "for this session" in capsys.readouterr().out


def test_mcp_remove_reports_a_bad_name(repl, client, mocker, capsys):
    client.remove_server = mocker.AsyncMock(side_effect=ValueError("Unknown MCP server 'nope'"))
    repl(["/mcp remove nope"])
    assert "Unknown MCP server" in capsys.readouterr().err


def test_mcp_remove_without_a_name_shows_usage(repl, client, capsys):
    repl(["/mcp remove"])
    assert "Usage: /mcp remove" in capsys.readouterr().err


# ---------------- /sessions, /delete, /compact ----------------

def test_sessions_command_lists_without_running_a_turn(repl, mocker):
    listed = mocker.patch.object(cli_mod, "_print_sessions")
    agent_run = repl(["/sessions"])
    listed.assert_called_once()
    agent_run.assert_not_awaited()


def test_delete_command_removes_a_session(repl, mocker, cfg):
    from omni.session_store import SessionStore
    sid = SessionStore(cfg.db_path).create_session("/p", "m", "t", name="doomed")
    repl(["/delete doomed"])
    assert not SessionStore(cfg.db_path).session_exists(sid)


def test_delete_unknown_session_reports_to_stderr(repl, capsys):
    repl(["/delete ghost"])
    assert "No session found" in capsys.readouterr().err


def test_compact_before_any_turn_says_so(repl, capsys):
    repl(["/compact"])
    assert "No active session" in capsys.readouterr().out


def test_compact_uses_the_resolved_session_id(mocker, client, cfg, capsys):
    """Resuming by --session-name must resolve to the real id up front, or
    /compact silently finds no messages."""
    from omni.session_store import SessionStore
    store = SessionStore(cfg.db_path)
    sid = store.create_session("/p", "m", "t", name="my-session")
    for i in range(30):
        store.append_message(sid, i, {"role": "user", "content": f"m{i}"})

    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch.object(cli_mod, "_show_resumed_history")
    mocker.patch.object(cli_mod, "_make_tui", lambda *a, **k: None)
    scripted(mocker, ["/compact", "/exit"])
    compact = mocker.patch.object(cli_mod.CodingAgent, "compact_history",
                                 mocker.AsyncMock(return_value="Compacted 30 down to 5"))
    mocker.patch("prompt_toolkit.PromptSession", mocker.Mock())
    mocker.patch("prompt_toolkit.patch_stdout.patch_stdout", mocker.MagicMock())
    asyncio.run(cli_mod._interactive(cfg, "my-session", None))
    assert compact.await_args.args[0] == sid          # real id, not the name


# ---------------- /model ----------------

def test_model_command_lists_models(mocker, client, cfg):
    """This asserted `pytest.raises(Exception)` and passed on the NameError
    that made /model unusable — the bug stayed green because the test was
    pinning the crash. It now asserts the command works."""
    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch.object(cli_mod, "list_models", mocker.AsyncMock(return_value=["a", "b"]))
    mocker.patch.object(cli_mod, "_make_tui", lambda *a, **k: None)
    scripted(mocker, ["/model", "/exit"])
    echo = mocker.patch.object(cli_mod, "_echo")
    asyncio.run(cli_mod._interactive(cfg, None, None))
    said = " ".join(str(c.args[0]) for c in echo.call_args_list if c.args)
    assert "a" in said and "b" in said


def test_model_name_command_switches_without_redrawing_the_header(repl, mocker, cfg, capsys):
    """The header is startup furniture: a /model switch echoes the new name
    instead of dropping a second copy of the box into the transcript."""
    header = mocker.patch.object(cli_mod, "_print_header")
    repl(["/model llama3.1:latest"])
    assert cfg.model == "llama3.1:latest"
    assert "llama3.1:latest" in capsys.readouterr().out
    assert header.call_count == 1          # the one at startup


# ---------------- settings ----------------

def test_a_setting_typed_at_the_repl_is_saved(repl, cfg, settings_path):
    repl(["/max-steps 7", "/context-window-budget 80000"])
    assert cfg.max_steps == 7 and cfg.context_window_budget == 80_000
    saved = json.loads(settings_path.read_text())
    assert saved == {"maxSteps": 7, "contextWindowBudget": 80_000}


def test_a_setting_is_reset_from_the_repl(repl, cfg, settings_path):
    settings_path.write_text(json.dumps({"maxSteps": 7}))
    cfg.max_steps = 7
    repl(["/max-steps reset"])
    assert cfg.max_steps == cli_mod.AgentConfig.max_steps
    assert json.loads(settings_path.read_text()) == {}


def test_config_lists_the_settings_at_the_repl(repl, capsys):
    repl(["/config"])
    out = capsys.readouterr().out
    assert "/max-steps" in out and "/system-prompt" in out


def test_an_unknown_slash_command_is_refused_not_run_as_a_task(repl, capsys):
    """It used to fall through and be dispatched as the task, so a near miss
    became a sentence for the agent to act on while the setting it meant
    never moved — the one failure mode where nothing looks wrong."""
    agent_run = repl(["/auto-aprove on"])
    agent_run.assert_not_awaited()
    err = capsys.readouterr().err
    assert "/auto-aprove isn't a command" in err and "/auto-approve" in err


def test_a_pasted_path_is_still_a_task(repl):
    """A leading slash is not by itself a command: "/etc/hosts has the wrong
    entry" is something to go and do."""
    agent_run = repl(["/etc/hosts has the wrong entry"])
    assert agent_run.await_args.args[0] == "/etc/hosts has the wrong entry"


def test_a_setting_is_reachable_however_its_name_is_spelled(repl, cfg, settings_path):
    """A hyphen is the separator nobody reaches for when the name reads as
    two words. All three spellings move the one saved value."""
    for typed, expected in (("/auto-approve on", True), ("/auto_approve off", False),
                            ("/auto approve on", True)):
        repl([typed])
        assert cfg.auto_approve is expected, typed
        assert json.loads(settings_path.read_text()) == {"autoApprove": expected}


def test_a_task_that_opens_with_a_setting_name_is_not_a_setting(repl, cfg):
    """Without the leading slash it is prose: "model the login flow" used to
    set the model to "the login flow" and save it for every later run."""
    agent_run = repl(["model the login flow in a diagram"])
    assert agent_run.await_args.args[0] == "model the login flow in a diagram"
    assert cfg.model == "test-model"


def test_header_is_not_redrawn_when_the_session_gets_its_id(repl, mocker):
    header = mocker.patch.object(cli_mod, "_print_header")
    mocker.patch.object(cli_mod.CodingAgent, "session_id", "abc12345", create=True)
    repl(["first task", "second task"])
    assert header.call_count == 1


def test_model_list_failure_is_reported(mocker, client, cfg, capsys):
    from omni.llm_client import LLMError
    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch.object(cli_mod, "list_models", mocker.AsyncMock(side_effect=LLMError("no /v1/models")))
    mocker.patch.object(cli_mod, "_make_tui", lambda *a, **k: None)
    scripted(mocker, ["/model", "/exit"])
    mocker.patch("prompt_toolkit.PromptSession", mocker.Mock())
    mocker.patch("prompt_toolkit.patch_stdout.patch_stdout", mocker.MagicMock())
    asyncio.run(cli_mod._interactive(cfg, None, None))
    assert "no /v1/models" in capsys.readouterr().err


# ---------------- /mcp and /mcp restart ----------------

def test_mcp_command_prints_status(repl, mocker):
    status = mocker.patch.object(cli_mod, "_print_mcp_status")
    repl(["/mcp"])
    status.assert_called()


def test_mcp_restart_reconnects_the_named_server(repl, client, capsys):
    client.restart_server.return_value = {
        "name": "docs", "connected": True, "error": None, "tool_count": 4}
    repl(["/mcp restart docs"])
    client.restart_server.assert_awaited_with("docs")
    assert "Restarted" in capsys.readouterr().out


def test_mcp_restart_all_hits_every_server(repl, client):
    client.server_names.return_value = ["built-in", "docs"]
    client.restart_server.return_value = {
        "name": "x", "connected": True, "error": None, "tool_count": 1}
    repl(["/mcp restart all"])
    assert {c.args[0] for c in client.restart_server.await_args_list} == {"built-in", "docs"}


def test_mcp_restart_reports_a_failed_reconnect(repl, client, capsys):
    client.restart_server.return_value = {
        "name": "docs", "connected": False, "error": "still broken", "tool_count": 0}
    repl(["/mcp restart docs"])
    assert "still broken" in capsys.readouterr().err


def test_mcp_restart_without_a_name_shows_usage(repl, client, capsys):
    repl(["/mcp restart"])
    client.restart_server.assert_not_awaited()
    assert "Usage" in capsys.readouterr().err


def test_mcp_restart_unknown_server_is_reported(repl, client, capsys):
    client.restart_server.side_effect = ValueError("Unknown MCP server 'ghost'")
    repl(["/mcp restart ghost"])
    assert "ghost" in capsys.readouterr().err


def test_mcp_restart_refreshes_prompt_completions(repl, client):
    client.restart_server.return_value = {
        "name": "docs", "connected": True, "error": None, "tool_count": 1}
    repl(["/mcp restart docs"])
    assert client.list_prompts.await_count >= 2      # re-listed after the restart


def test_mcp_tools_lists_one_servers_tools(repl, client, mocker):
    client.server_tools.return_value = [
        {"name": "docs__search", "real_name": "search", "description": "Search",
         "deferred": False, "revealed": False, "internal": False}]
    printed = mocker.patch.object(cli_mod, "_print_server_tools")
    repl(["/mcp tools docs"])
    client.server_tools.assert_awaited_with("docs")
    assert printed.call_args.args[0] == "docs"
    assert printed.call_args.args[1][0]["name"] == "docs__search"


def test_mcp_tools_without_a_name_shows_usage(repl, client, capsys):
    repl(["/mcp tools"])
    client.server_tools.assert_not_awaited()
    err = capsys.readouterr().err
    assert "Usage" in err and "built-in" in err        # lists valid names


def test_mcp_tools_unknown_server_is_reported(repl, client, capsys):
    client.server_tools.side_effect = ValueError("Unknown MCP server 'ghost'")
    repl(["/mcp tools ghost"])
    assert "ghost" in capsys.readouterr().err


def test_mcp_tools_unconnected_server_is_reported(repl, client, capsys):
    client.server_tools.side_effect = ValueError("MCP server 'docs' isn't connected")
    repl(["/mcp tools docs"])
    assert "isn't connected" in capsys.readouterr().err


def test_mcp_tools_does_not_run_a_turn(repl, client, mocker):
    client.server_tools.return_value = []
    mocker.patch.object(cli_mod, "_print_server_tools")
    agent_run = repl(["/mcp tools docs"])
    agent_run.assert_not_awaited()


def test_bare_mcp_still_shows_status_not_tools(repl, client, mocker):
    """The `/mcp tools` branch must not swallow the plain `/mcp` command."""
    status = mocker.patch.object(cli_mod, "_print_mcp_status")
    tools = mocker.patch.object(cli_mod, "_print_server_tools")
    repl(["/mcp"])
    status.assert_called()
    tools.assert_not_called()


def test_mcp_tools_completions_are_registered_per_server(repl, client, mocker):
    client.server_names.return_value = ["built-in", "docs"]
    repl([])
    assert "/mcp tools built-in" in repl.captured["commands"]
    assert "/mcp tools docs" in repl.captured["commands"]


# ---------------- /resources ----------------

def test_resources_command_lists(repl, client, mocker):
    client.list_resources.return_value = {
        "file:///a.md": {"server": "docs", "name": "n", "description": "d",
                         "mime_type": "text/markdown", "size": 1,
                         "template": False, "shadowed_by": []}}
    printed = mocker.patch.object(cli_mod, "_print_resources")
    repl(["/resources"])
    printed.assert_called_once()


def test_resources_command_reads_one(repl, client, mocker):
    client.list_resources.return_value = {
        "file:///a.md": {"server": "docs", "name": "", "description": "",
                         "mime_type": "", "size": None,
                         "template": False, "shadowed_by": []}}
    client.read_resource.return_value = "the contents"
    shown = mocker.patch("omni.ui.resource_content")
    repl(["/resources file:///a.md"])
    client.read_resource.assert_awaited_with("file:///a.md")
    shown.assert_called_once_with("file:///a.md", "the contents")


def test_resources_read_failure_is_reported(repl, client, capsys):
    client.list_resources.return_value = {}
    client.read_resource.side_effect = ValueError("Unknown resource")
    repl(["/resources x://nope"])
    assert "Unknown resource" in capsys.readouterr().err


def test_resources_list_failure_is_reported(repl, client, capsys):
    client.list_resources.side_effect = RuntimeError("transport died")
    repl(["/resources"])
    assert "transport died" in capsys.readouterr().err


# ---------------- /btw ----------------

def test_btw_answers_without_running_a_turn(repl, mocker):
    handled = mocker.patch.object(cli_mod, "_handle_btw", mocker.AsyncMock())
    agent_run = repl(["/btw what is a decorator?"])
    handled.assert_awaited_once()
    assert handled.await_args.args[1] == "what is a decorator?"
    agent_run.assert_not_awaited()


def test_btw_without_a_question_does_nothing(repl, mocker):
    handled = mocker.patch.object(cli_mod, "_handle_btw", mocker.AsyncMock())
    repl(["/btw", "/btw    "])
    handled.assert_not_awaited()


# ---------------- MCP prompts as slash commands ----------------

def test_mcp_prompt_is_resolved_and_run_as_the_task(repl, client, mocker):
    client.list_prompts.return_value = {
        "docs:summarize": {"description": "Summarize a file",
                           "arguments": [{"name": "path", "description": "", "required": True}]}}
    client.get_prompt.return_value = "Please summarize README.md"
    agent_run = repl(["/docs:summarize README.md"])
    client.get_prompt.assert_awaited_with("docs:summarize", {"path": "README.md"})
    assert agent_run.await_args.args[0] == "Please summarize README.md"


def test_mcp_prompt_quoted_argument_is_kept_whole(repl, client):
    client.list_prompts.return_value = {
        "docs:search": {"description": "", "arguments": [
            {"name": "q", "description": "", "required": True}]}}
    client.get_prompt.return_value = "resolved"
    repl(['/docs:search "two words"'])
    assert client.get_prompt.await_args.args[1] == {"q": "two words"}


def test_mcp_prompt_missing_required_argument_is_refused(repl, client, capsys):
    client.list_prompts.return_value = {
        "docs:summarize": {"description": "", "arguments": [
            {"name": "path", "description": "", "required": True}]}}
    agent_run = repl(["/docs:summarize"])
    client.get_prompt.assert_not_awaited()
    assert "missing required argument" in capsys.readouterr().err
    agent_run.assert_not_awaited()


def test_mcp_prompt_too_many_arguments_is_refused(repl, client, capsys):
    client.list_prompts.return_value = {
        "docs:p": {"description": "", "arguments": [
            {"name": "a", "description": "", "required": True}]}}
    repl(["/docs:p one two three"])
    client.get_prompt.assert_not_awaited()
    assert "at most 1 argument" in capsys.readouterr().err


def test_mcp_prompt_unbalanced_quotes_are_reported(repl, client, capsys):
    client.list_prompts.return_value = {
        "docs:p": {"description": "", "arguments": [
            {"name": "a", "description": "", "required": False}]}}
    repl(['/docs:p "unclosed'])
    assert "Error parsing arguments" in capsys.readouterr().err


def test_mcp_prompt_resolution_failure_is_reported(repl, client, capsys):
    client.list_prompts.return_value = {
        "docs:p": {"description": "", "arguments": []}}
    client.get_prompt.side_effect = RuntimeError("prompt blew up")
    agent_run = repl(["/docs:p"])
    assert "prompt blew up" in capsys.readouterr().err
    agent_run.assert_not_awaited()


def test_prompt_commands_are_registered_for_completion(repl, client, mocker):
    client.list_prompts.return_value = {
        "docs:summarize": {"description": "Summarize", "arguments": [
            {"name": "path", "description": "", "required": True},
            {"name": "style", "description": "", "required": False}]}}
    repl([])
    entry = repl.captured["commands"]["/docs:summarize "]
    assert "Summarize" in entry and "<path>" in entry and "[style]" in entry


def test_a_slash_name_no_prompt_answers_to_is_refused(repl, client, capsys):
    """Once the MCP prompts have had their look, nothing else can claim the
    line — so say so rather than handing it to the agent as a task."""
    client.list_prompts.return_value = {}
    agent_run = repl(["/not-a-command at all"])
    agent_run.assert_not_awaited()
    assert "/not-a-command isn't a command" in capsys.readouterr().err


# ---------------- startup wiring ----------------

def test_failed_server_warning_is_shown_at_startup(repl, client, capsys):
    client.server_status.return_value = [
        {"name": "docs", "connected": False, "connected_for": None, "error": "boom",
         "deferred": False, "tool_count": 0, "target": "x"}]
    repl([])
    assert "failed to connect" in capsys.readouterr().err


def test_resumed_history_is_shown_once(mocker, client, cfg):
    mocker.patch.object(cli_mod, "_print_header")
    shown = mocker.patch.object(cli_mod, "_show_resumed_history")
    mocker.patch.object(cli_mod, "_make_tui", lambda *a, **k: None)
    scripted(mocker, ["/exit"])
    mocker.patch("prompt_toolkit.PromptSession", mocker.Mock())
    mocker.patch("prompt_toolkit.patch_stdout.patch_stdout", mocker.MagicMock())
    asyncio.run(cli_mod._interactive(cfg, "some-session", None))
    shown.assert_called_once()


def test_model_completions_are_registered_when_available(repl, client, mocker):
    mocker.patch.object(cli_mod, "list_models", mocker.AsyncMock(return_value=["m1", "m2"]))
    repl([])
    assert "/model m1" in repl.captured["commands"] and "/model m2" in repl.captured["commands"]


def test_missing_v1_models_endpoint_is_tolerated(repl, client, mocker):
    from omni.llm_client import LLMError
    mocker.patch.object(cli_mod, "list_models", mocker.AsyncMock(side_effect=LLMError("404")))
    repl(["a task"])            # startup must not fail


# ---------------- subagents ----------------
#
# These drive cli._run_turn against a real TuiApp, because that pairing is
# where the interesting mistakes live: a subagent's output has to land in its
# own pane, and a missing attribute on Pane silently turned every spawn into
# an error result the model then worked around.

@pytest.fixture
def tui_app(mocker, cfg, headless_terminal):
    from omni import ui
    from omni.tui import TuiApp
    app = TuiApp({}, "main", cfg.model)
    ui.use_tui(app)
    yield app
    ui.use_tui(None)


def _stub_agent(mocker, label: str):
    """An agent whose run() emits the way the real one does."""
    from omni import ui

    async def run(task, **kwargs):
        ui.assistant_message(f"{label}: {task}")
        return f"{label} finished"

    agent = mocker.Mock(session_id=f"sess-{label}", tokens={"prompt": 10, "completion": 5})
    agent.run = mocker.AsyncMock(side_effect=run)
    return agent


async def test_a_subagents_output_lands_in_its_own_pane(tui_app, mocker, cfg):
    tui_app.main.agent = _stub_agent(mocker, "main")
    sub = tui_app.add_pane("child", agent=_stub_agent(mocker, "child"), depth=1)

    await cli_mod._run_turn(tui_app.main, "parent work", cfg=cfg, client=mocker.Mock(),
                            tui=tui_app, session_name=None)
    await cli_mod._run_turn(sub, "child work", cfg=cfg, client=mocker.Mock(),
                            tui=tui_app, session_name=None)

    def texts(pane):
        return " ".join(b.collapsed() for b in pane.transcript.blocks)

    assert "main: parent work" in texts(tui_app.main)
    assert "child: child work" in texts(sub)
    assert "child: child work" not in texts(tui_app.main)


async def test_a_pane_carries_its_own_session(tui_app, mocker, cfg):
    """Every pane resumes its own session; a Pane without this attribute made
    each spawn fail with an AttributeError the model just worked around."""
    sub = tui_app.add_pane("child", agent=_stub_agent(mocker, "child"), depth=1)
    assert sub.session_id is None
    await cli_mod._run_turn(sub, "work", cfg=cfg, client=mocker.Mock(), tui=tui_app,
                            session_name=None)
    assert sub.session_id == "sess-child"
    assert sub.agent.run.await_args.kwargs["resume_session_id"] is None


async def test_a_finished_subagent_is_marked_done(tui_app, mocker, cfg):
    sub = tui_app.add_pane("child", agent=_stub_agent(mocker, "child"), depth=1)
    await cli_mod._run_turn(sub, "work", cfg=cfg, client=mocker.Mock(), tui=tui_app,
                            session_name=None)
    tui_app.set_idle(pane=sub, done=True)
    assert sub.done is True and "●" in "".join(f[1] for f in tui_app._agent_tree())


async def test_the_spawner_reports_a_failure_instead_of_raising(tui_app, mocker, cfg):
    """A broken subagent is the parent's business to hear about, not a reason
    to break the parent's turn."""
    mocker.patch.object(cli_mod, "_run_turn", mocker.AsyncMock(side_effect=RuntimeError("boom")))
    cli_mod._RUNTIME["client"] = mocker.Mock()
    spawn = cli_mod._make_spawner(cfg, tui_app, tui_app.main)
    out = await spawn("do a thing", name="thing")
    assert "failed to run" in out and "boom" in out


async def test_a_subagent_cannot_spawn_another(tui_app, mocker, cfg):
    from omni.tui import current_pane
    cli_mod._RUNTIME["client"] = mocker.Mock()
    sub = tui_app.add_pane("child", agent=_stub_agent(mocker, "child"), depth=1)
    spawn = cli_mod._make_spawner(cfg, tui_app, tui_app.main)
    token = current_pane.set(sub)
    try:
        out = await spawn("go deeper")
    finally:
        current_pane.reset(token)
    assert "can't spawn subagents" in out


async def test_the_spawner_gives_the_child_its_own_config(tui_app, mocker, cfg):
    """A subagent's model, prompt and step budget are its own — that's what
    --subagent-model and spawn_agent's system_prompt are for."""
    cfg.subagent_model = "small-model"
    cfg.subagent_max_steps = 7
    cli_mod._RUNTIME["client"] = mocker.Mock()
    captured = {}

    async def fake_turn(pane, task, *, cfg, **kwargs):
        captured["cfg"] = cfg
        captured["pane"] = pane
        return "ok"

    mocker.patch.object(cli_mod, "_run_turn", fake_turn)
    spawn = cli_mod._make_spawner(cfg, tui_app, tui_app.main)
    assert await spawn("audit it", name="audit", system_prompt="You are a REVIEWER.") == "ok"
    child_cfg = captured["cfg"]
    assert child_cfg.model == "small-model"
    assert child_cfg.system_prompt == "You are a REVIEWER."
    assert child_cfg.max_steps == 7
    assert captured["pane"].name == "audit" and captured["pane"].kind == "subagent"
    assert cfg.model != "small-model"        # the parent's own config is untouched


async def test_an_interrupted_subagent_says_so_to_its_parent(tui_app, mocker, cfg):
    """"It didn't finish" isn't enough: a model told only that tends to spawn
    the same job again."""
    cli_mod._RUNTIME["client"] = mocker.Mock()

    async def interrupted_turn(pane, task, **kwargs):
        pane.outcome, pane.error = "interrupted", ""
        return None

    mocker.patch.object(cli_mod, "_run_turn", interrupted_turn)
    spawn = cli_mod._make_spawner(cfg, tui_app, tui_app.main)
    out = await spawn("long job", name="slow")
    assert "interrupted by the user" in out and "Don't spawn it again" in out


async def test_a_failed_subagent_reports_the_reason(tui_app, mocker, cfg):
    cli_mod._RUNTIME["client"] = mocker.Mock()

    async def failed_turn(pane, task, **kwargs):
        pane.outcome, pane.error = "error", "LLM server unreachable"
        return None

    mocker.patch.object(cli_mod, "_run_turn", failed_turn)
    spawn = cli_mod._make_spawner(cfg, tui_app, tui_app.main)
    out = await spawn("job", name="broken")
    assert "failed: LLM server unreachable" in out


async def test_interrupting_a_turn_stops_the_agent_too(tui_app, mocker, cfg):
    """Cancelling the await alone would leave the agent running in the
    background, still spending tokens on a turn nobody is waiting for."""
    started = asyncio.Event()

    async def never_ends(task, **kwargs):
        started.set()
        await asyncio.sleep(30)

    agent = mocker.Mock(session_id="s", tokens={"prompt": 0, "completion": 0})
    agent.run = mocker.AsyncMock(side_effect=never_ends)
    pane = tui_app.add_pane("slow", agent=agent, depth=1)

    turn = asyncio.ensure_future(cli_mod._run_turn(pane, "work", cfg=cfg,
                                                    client=mocker.Mock(), tui=tui_app,
                                                    session_name=None))
    await started.wait()
    inner = pane.task
    turn.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await turn
    await asyncio.sleep(0)
    assert inner.cancelled() or inner.done()
    assert pane.outcome in ("interrupted", "")


async def test_an_interrupted_subagent_is_marked_in_the_tree(tui_app, mocker, cfg):
    sub = tui_app.add_pane("slow", agent=_stub_agent(mocker, "slow"), depth=1)
    sub.outcome = "interrupted"
    rows = "".join(f[1] for f in tui_app._agent_tree())
    assert "✕" in rows and "interrupted" in rows


# ---------------- /copy ----------------


def test_copy_without_a_full_screen_app_points_at_the_scrollback(repl, capsys):
    repl(["/copy"])
    assert "scrollback" in capsys.readouterr().out


def test_copy_hands_the_transcript_to_the_clipboard(mocker, client, cfg):
    """End to end: a real TuiApp, running over a pipe, and "/copy" typed at it.

    The app draws the transcript itself, so the terminal has no copy of it to
    select from — this is the path that gets one out."""
    import io

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output
    from rich.text import Text

    from omni import ui
    from omni.tui import TuiApp

    copy = mocker.patch("omni.clipboard.copy_text", return_value=True)
    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch("omni.ui.final_result")
    apps = []

    def fake_make_tui(commands, session_label, model):
        app = TuiApp(commands, session_label, model)
        app.emit(lambda: Text("what the model said"))
        ui.use_tui(app)
        apps.append(app)
        return app

    mocker.patch.object(cli_mod, "_make_tui", fake_make_tui)
    scripted(mocker, ["/copy", "/exit"])

    with create_pipe_input() as pipe:
        output = Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=100),
                               term="xterm-256color")
        with create_app_session(input=pipe, output=output):
            asyncio.run(cli_mod._interactive(cfg, None, None))

    assert "what the model said" in copy.call_args.args[0]
    # The confirmation goes into the transcript, since the app owns the screen.
    # Blocks render lazily against the registered app, so put it back first.
    ui.use_tui(apps[0])
    try:
        rendered = apps[0].main.transcript.plain_text(80)
    finally:
        ui.use_tui(None)
    assert "Copied 1 line to the clipboard." in rendered


# ---------------- the fallback queue behind a pane ----------------
#
# The normal home for a line typed mid-turn is the running agent's inbox
# (see "routing a line typed mid-turn" below). Pane.pending is what catches
# the rest — a line typed with no agent to hand it to, and one the turn
# ended before it could fold in. Those run as turns of their own, and this
# is the guarantee about how: strictly in order, one at a time, never
# concurrently with the turn already running and never at another agent.


@pytest.fixture
async def dispatcher(tui_app, mocker, cfg):
    """_interactive's _start_turn / _turn_finished pair, over a _run_turn
    stand-in that finishes only when told.

    Rebuilt here rather than driven through the REPL loop because the
    ordering is the thing under test, and a scripted REPL would be choosing
    the interleaving for us."""
    order = []
    gates = {}
    live = []

    async def fake_run_turn(pane, task, *, on_finished=None, **kwargs):
        order.append((pane.name, task))
        gate = gates.setdefault(task, asyncio.Event())
        pane.task = asyncio.current_task()
        try:
            await gate.wait()
        finally:
            pane.task = None
            if on_finished is not None:
                on_finished(pane)

    mocker.patch.object(cli_mod, "_run_turn", fake_run_turn)
    mocker.patch.object(cli_mod, "_echo")

    def turn_finished(pane):
        if pane.pending:
            text, attachments = pane.pending.pop(0)
            start(pane, text, attachments)

    def start(pane, text, attachments=None):
        if attachments is None:
            attachments, pane.attachments = pane.attachments, []
        if pane.task is not None and not pane.task.done():
            pane.pending.append((text, attachments))
            return
        pane.task = asyncio.ensure_future(
            cli_mod._run_turn(pane, text, cfg=cfg, client=mocker.Mock(), tui=tui_app,
                               session_name=None, on_finished=turn_finished,
                               attachments=attachments))
        live.append(pane.task)

    async def finish(task):
        """Let the named turn complete, and let its successor start."""
        for _ in range(50):
            if task in gates:
                break
            await asyncio.sleep(0)
        gates.setdefault(task, asyncio.Event()).set()
        for _ in range(4):
            await asyncio.sleep(0)

    start.order = order
    start.finish = finish
    yield start

    # Nothing may outlive the test. A gate still being awaited would be
    # cancelled as the loop closes, and its `finally` would then try to
    # dispatch the next queued line with no loop left to dispatch onto — so
    # the queues are emptied first and the tasks cancelled here, while this
    # (async, hence in-loop) fixture still has a loop to do it on.
    for pane in tui_app.panes:
        pane.pending.clear()
    for task in live:
        task.cancel()
    await asyncio.gather(*live, return_exceptions=True)


async def test_a_line_typed_while_busy_waits_instead_of_running_alongside(dispatcher, tui_app):
    pane = tui_app.main
    dispatcher(pane, "first")
    await asyncio.sleep(0)
    dispatcher(pane, "second")
    assert dispatcher.order == [("main", "first")]          # the second hasn't started
    assert [text for text, _ in pane.pending] == ["second"]


async def test_queued_lines_run_in_the_order_they_were_typed(dispatcher, tui_app):
    """Sequential, not parallel and not reversed: the queue is a FIFO and
    only one turn per pane is ever in flight."""
    pane = tui_app.main
    dispatcher(pane, "first")
    await asyncio.sleep(0)
    for text in ("second", "third", "fourth"):
        dispatcher(pane, text)

    for text in ("first", "second", "third"):
        await dispatcher.finish(text)
    await dispatcher.finish("fourth")

    assert dispatcher.order == [("main", t) for t in
                                 ("first", "second", "third", "fourth")]
    assert pane.pending == []


async def test_only_one_turn_per_pane_is_ever_in_flight(dispatcher, tui_app):
    pane = tui_app.main
    dispatcher(pane, "first")
    await asyncio.sleep(0)
    dispatcher(pane, "second")
    dispatcher(pane, "third")
    # Three lines sent, one running, two waiting.
    assert len(dispatcher.order) == 1 and len(pane.pending) == 2


async def test_each_agent_drains_its_own_queue(dispatcher, tui_app):
    """Two agents working at once, each with its own backlog — neither line
    may run at the other's agent."""
    main = tui_app.main
    sub = tui_app.add_pane("audit", depth=1)
    dispatcher(main, "main first")
    dispatcher(sub, "sub first")
    await asyncio.sleep(0)
    dispatcher(main, "main second")
    dispatcher(sub, "sub second")

    assert [t for t, _ in main.pending] == ["main second"]
    assert [t for t, _ in sub.pending] == ["sub second"]

    await dispatcher.finish("sub first")
    assert dispatcher.order[-1] == ("audit", "sub second")
    assert [t for t, _ in main.pending] == ["main second"]   # untouched

    await dispatcher.finish("main first")
    assert dispatcher.order[-1] == ("main", "main second")


async def test_images_pasted_with_a_queued_line_stay_with_that_line(dispatcher, tui_app):
    """The attachments are taken when the line is typed, not when its turn
    finally runs — otherwise a screenshot pasted for the second question
    would arrive with the third."""
    pane = tui_app.main
    dispatcher(pane, "first")
    await asyncio.sleep(0)
    pane.attachments.append({"mime": "image/png", "data": b"shot"})
    dispatcher(pane, "look at this")
    assert pane.attachments == []                       # taken with the line
    assert pane.pending[0][1] == [{"mime": "image/png", "data": b"shot"}]


async def test_a_turn_that_raises_still_lets_the_queue_move_on(tui_app, mocker, cfg):
    """on_finished sits in the real _run_turn's `finally`, so a turn that
    fails must not strand everything typed behind it."""
    from omni.llm_client import LLMError
    agent = mocker.Mock(session_id="s", tokens={"prompt": 0, "completion": 0})
    agent.run = mocker.AsyncMock(side_effect=LLMError("the server fell over"))
    pane = tui_app.add_pane("flaky", agent=agent, depth=1)
    pane.pending.append(("still my turn next", []))

    drained = []
    await cli_mod._run_turn(pane, "explodes", cfg=cfg, client=mocker.Mock(),
                             tui=tui_app, session_name=None,
                             on_finished=lambda p: drained.append(p.pending.pop(0)[0]))
    assert drained == ["still my turn next"]
    assert pane.outcome == "error"


# ---------------- what the terminal tab says ----------------


async def test_the_tab_keeps_the_session_name_after_a_turn(tui_app, mocker, cfg):
    """_run_turn renames the tab after every main turn. It used to set the
    raw id, so a session was called "my refactor" until its first turn landed
    and a hex blob from then on."""
    from omni.session_store import SessionStore
    store = SessionStore(cfg.db_path)
    sid = store.create_session(cfg.project_root, cfg.model, "earlier", name="my refactor")

    agent = mocker.Mock(session_id=sid, store=store, tokens={"prompt": 0, "completion": 0})
    agent.run = mocker.AsyncMock(return_value="done")
    tui_app.main.agent = agent
    title = mocker.patch("omni.ui.set_terminal_title")

    await cli_mod._run_turn(tui_app.main, "carry on", cfg=cfg, client=mocker.Mock(),
                             tui=tui_app, session_name=None)
    assert title.call_args.args[0] == "my refactor"


async def test_an_unnamed_session_still_titles_the_tab_with_its_id(tui_app, mocker, cfg):
    from omni.session_store import SessionStore
    store = SessionStore(cfg.db_path)
    sid = store.create_session(cfg.project_root, cfg.model, "earlier")

    agent = mocker.Mock(session_id=sid, store=store, tokens={"prompt": 0, "completion": 0})
    agent.run = mocker.AsyncMock(return_value="done")
    tui_app.main.agent = agent
    title = mocker.patch("omni.ui.set_terminal_title")

    await cli_mod._run_turn(tui_app.main, "carry on", cfg=cfg, client=mocker.Mock(),
                             tui=tui_app, session_name=None)
    assert title.call_args.args[0] == sid


async def test_a_subagents_turn_never_renames_the_tab(tui_app, mocker, cfg):
    """The tab belongs to the session you are in, not to whatever a
    background agent happened to finish."""
    agent = mocker.Mock(session_id="child", tokens={"prompt": 0, "completion": 0})
    agent.run = mocker.AsyncMock(return_value="done")
    sub = tui_app.add_pane("audit", agent=agent, depth=1)
    title = mocker.patch("omni.ui.set_terminal_title")

    await cli_mod._run_turn(sub, "look", cfg=cfg, client=mocker.Mock(),
                             tui=tui_app, session_name=None)
    title.assert_not_called()


async def test_a_store_that_cannot_answer_does_not_break_the_turns_tail(tui_app, mocker, cfg):
    """The id is always a usable answer, so naming the tab must never raise
    into the end of an otherwise successful turn."""
    agent = mocker.Mock(session_id="sid", tokens={"prompt": 0, "completion": 0})
    agent.store.display_name.side_effect = RuntimeError("database is locked")
    agent.run = mocker.AsyncMock(return_value="done")
    tui_app.main.agent = agent
    title = mocker.patch("omni.ui.set_terminal_title")

    result = await cli_mod._run_turn(tui_app.main, "carry on", cfg=cfg, client=mocker.Mock(),
                                      tui=tui_app, session_name=None)
    assert result == "done"
    assert title.call_args.args[0] == "sid"


# ---------------- routing a line typed mid-turn ----------------
#
# A line typed while an agent works goes to the agent running that turn, so
# the model acts on it during the turn. Only what no turn could take — no
# agent yet, or a turn that ended before the fold-in came round — falls back
# to running as a turn of its own.


class _Inbox:
    """A stand-in agent with the inbox contract _start_turn relies on."""

    def __init__(self, accepts=True, raises=False):
        self.taken = []
        self.accepts = accepts
        self.raises = raises

    def inject(self, text, attachments=None):
        if self.raises:
            raise RuntimeError("the turn just tore down")
        if not self.accepts:
            return False
        self.taken.append((text, list(attachments or [])))
        return True

    @property
    def inbox_depth(self):
        return len(self.taken)

    def take_inbox(self):
        left, self.taken = self.taken, []
        return left


def test_a_line_typed_at_a_working_agent_is_handed_to_it(tui_app):
    pane = tui_app.main
    pane.agent = _Inbox()
    assert cli_mod._inject(pane, "also check the tests", []) is True
    assert pane.agent.taken == [("also check the tests", [])]


def test_a_pane_with_no_agent_takes_nothing(tui_app):
    tui_app.main.agent = None
    assert cli_mod._inject(tui_app.main, "hello", []) is False


def test_an_agent_that_predates_the_inbox_takes_nothing(tui_app, mocker):
    """An older stub has no inject at all; the line must fall back rather
    than raise into the REPL."""
    tui_app.main.agent = object()
    assert cli_mod._inject(tui_app.main, "hello", []) is False


def test_an_agent_that_refuses_is_believed(tui_app):
    tui_app.main.agent = _Inbox(accepts=False)
    assert cli_mod._inject(tui_app.main, "hello", []) is False


def test_an_agent_that_raises_does_not_take_the_repl_down(tui_app):
    """A turn tearing down mid-handover is not a reason to lose the line —
    the caller parks it instead."""
    tui_app.main.agent = _Inbox(raises=True)
    assert cli_mod._inject(tui_app.main, "hello", []) is False


def test_only_an_exact_true_counts_as_taken(tui_app, mocker):
    """A Mock answers every call with something truthy. Believing one would
    drop the line on the floor, which is the one outcome not allowed."""
    tui_app.main.agent = mocker.Mock()
    assert cli_mod._inject(tui_app.main, "hello", []) is False


def test_taking_the_inbox_empties_it(tui_app):
    pane = tui_app.main
    pane.agent = _Inbox()
    pane.agent.inject("one")
    pane.agent.inject("two")
    assert [text for text, _ in cli_mod._take_inbox(pane)] == ["one", "two"]
    assert cli_mod._take_inbox(pane) == []


@pytest.mark.parametrize("agent", [None, object(), "not an agent"])
def test_taking_from_something_that_is_not_an_agent_is_empty(tui_app, agent):
    tui_app.main.agent = agent
    assert cli_mod._take_inbox(tui_app.main) == []


def test_taking_from_a_stub_that_answers_oddly_is_empty(tui_app, mocker):
    """This runs in a turn's `finally`: a stub whose answer is not iterable
    must not turn a finished turn into an exception."""
    tui_app.main.agent = mocker.Mock()          # take_inbox() -> a Mock
    assert cli_mod._take_inbox(tui_app.main) == []


async def test_what_the_turn_never_read_is_moved_to_the_panes_queue(tui_app, mocker, cfg):
    """Typed on the turn's last step, so it arrived too late to be folded in.
    Late is recoverable; silently dropped is not."""
    inbox = _Inbox()
    agent = mocker.Mock(session_id="s", tokens={"prompt": 0, "completion": 0})
    agent.run = mocker.AsyncMock(return_value="done")
    agent.inject = inbox.inject
    agent.take_inbox = inbox.take_inbox
    agent.inbox_depth = 0
    pane = tui_app.add_pane("worker", agent=agent, depth=1)
    inbox.inject("arrived too late")

    await cli_mod._run_turn(pane, "go", cfg=cfg, client=mocker.Mock(),
                             tui=tui_app, session_name=None)

    assert [text for text, _ in pane.pending] == ["arrived too late"]


async def test_a_late_line_goes_in_front_of_anything_already_parked(tui_app, mocker, cfg):
    """It was typed before whatever is already sitting on the queue."""
    inbox = _Inbox()
    agent = mocker.Mock(session_id="s", tokens={"prompt": 0, "completion": 0})
    agent.run = mocker.AsyncMock(return_value="done")
    agent.inject = inbox.inject
    agent.take_inbox = inbox.take_inbox
    agent.inbox_depth = 0
    pane = tui_app.add_pane("worker", agent=agent, depth=1)
    pane.pending.append(("parked later", []))
    inbox.inject("typed earlier")

    await cli_mod._run_turn(pane, "go", cfg=cfg, client=mocker.Mock(),
                             tui=tui_app, session_name=None)

    assert [text for text, _ in pane.pending] == ["typed earlier", "parked later"]


async def test_a_turn_with_nothing_outstanding_leaves_the_queue_alone(tui_app, mocker, cfg):
    agent = mocker.Mock(session_id="s", tokens={"prompt": 0, "completion": 0})
    agent.run = mocker.AsyncMock(return_value="done")
    agent.take_inbox = lambda: []
    pane = tui_app.add_pane("worker", agent=agent, depth=1)

    await cli_mod._run_turn(pane, "go", cfg=cfg, client=mocker.Mock(),
                             tui=tui_app, session_name=None)
    assert pane.pending == []


def test_each_agent_is_handed_only_what_was_typed_at_it(tui_app):
    """Multi-agent: the inbox is the agent's, so a line typed at a working
    subagent joins that subagent's turn and nobody else's."""
    main, sub = tui_app.main, tui_app.add_pane("audit", depth=1)
    main.agent, sub.agent = _Inbox(), _Inbox()

    cli_mod._inject(main, "main thing", [])
    cli_mod._inject(sub, "sub thing", [])
    cli_mod._inject(sub, "another sub thing", [])

    assert [t for t, _ in main.agent.taken] == ["main thing"]
    assert [t for t, _ in sub.agent.taken] == ["sub thing", "another sub thing"]
    assert main.waiting == 1 and sub.waiting == 2


def test_images_travel_with_the_line_that_was_typed_with_them(tui_app):
    pane = tui_app.main
    pane.agent = _Inbox()
    shot = [{"mime": "image/png", "data": b"x"}]
    cli_mod._inject(pane, "what is wrong here", shot)
    assert pane.agent.taken == [("what is wrong here", shot)]


# ---------------- /cost ----------------


def test_cost_reports_every_pane_that_has_an_agent(tui_app, mocker, cfg):
    main, sub = tui_app.main, tui_app.add_pane("audit", depth=1)
    main.agent = mocker.Mock(tokens={"prompt": 100, "completion": 20})
    sub.agent = mocker.Mock(tokens={"prompt": 300, "completion": 40})
    report = mocker.patch("omni.ui.cost_report")

    cli_mod._report_cost(cfg, tui_app, main)
    rows = report.call_args.args[0]
    assert rows == [("main", 100, 20), ("1 audit", 300, 40)]


def test_cost_passes_the_cap_through(tui_app, mocker, cfg):
    cfg.max_turn_tokens = 12345
    tui_app.main.agent = mocker.Mock(tokens={"prompt": 1, "completion": 1})
    report = mocker.patch("omni.ui.cost_report")
    cli_mod._report_cost(cfg, tui_app, tui_app.main)
    assert report.call_args.args[1] == 12345


def test_a_pane_with_no_agent_is_left_out(tui_app, mocker, cfg):
    tui_app.main.agent = None
    tui_app.add_pane("audit", agent=mocker.Mock(tokens={"prompt": 5, "completion": 5}), depth=1)
    report = mocker.patch("omni.ui.cost_report")
    cli_mod._report_cost(cfg, tui_app, tui_app.main)
    assert [name for name, _, _ in report.call_args.args[0]] == ["1 audit"]


def test_an_agent_whose_tokens_are_not_a_count_is_left_out(tui_app, mocker, cfg):
    """A stub answers every attribute with a Mock; putting one in a table
    would render it as an object."""
    tui_app.main.agent = mocker.Mock()
    report = mocker.patch("omni.ui.cost_report")
    mocker.patch.object(cli_mod, "_echo")
    cli_mod._report_cost(cfg, tui_app, tui_app.main)
    report.assert_not_called()


def test_cost_before_anything_has_run_says_so(tui_app, mocker, cfg):
    tui_app.main.agent = None
    echo = mocker.patch.object(cli_mod, "_echo")
    cli_mod._report_cost(cfg, tui_app, tui_app.main)
    assert "Nothing spent yet" in echo.call_args.args[0]


# ---------------- commands that must not run mid-turn ----------------
#
# Typing while an agent works is ordinary now, which makes a few commands
# reachable at a moment they were never reachable before: the ones that
# rewrite or remove the session a running turn is still writing to.


def running_pane(tui_app, mocker, session_id="s1", name=None):
    pane = tui_app.main if name is None else tui_app.add_pane(name, depth=1)
    pane.session_id = session_id
    pane.task = mocker.Mock()
    pane.task.done.return_value = False
    return pane


def test_a_pane_with_a_live_turn_is_recognised(tui_app, mocker):
    pane = running_pane(tui_app, mocker)
    assert cli_mod._turn_in_flight(pane) is True


def test_a_finished_turn_is_not_in_flight(tui_app, mocker):
    pane = tui_app.main
    pane.task = mocker.Mock()
    pane.task.done.return_value = True
    assert cli_mod._turn_in_flight(pane) is False


def test_a_pane_that_never_ran_is_not_in_flight(tui_app):
    assert cli_mod._turn_in_flight(tui_app.main) is False


def test_something_without_a_task_attribute_is_not_in_flight():
    assert cli_mod._turn_in_flight(object()) is False


def test_the_pane_working_on_a_session_is_found_by_name(tui_app, mocker):
    """Every pane, not only the one the command was typed at: a subagent's
    session can be named on a /delete typed at main."""
    sub = running_pane(tui_app, mocker, session_id="child", name="audit")
    store = mocker.Mock()
    store.resolve_session_id.return_value = "child"
    assert cli_mod._pane_working_on(tui_app, tui_app.main, store, "whatever") == sub.name


def test_no_pane_is_working_on_an_idle_session(tui_app, mocker):
    tui_app.main.session_id = "s1"
    store = mocker.Mock()
    store.resolve_session_id.return_value = "s1"
    assert cli_mod._pane_working_on(tui_app, tui_app.main, store, "s1") is None


def test_a_name_that_resolves_to_nothing_blocks_nobody(tui_app, mocker):
    running_pane(tui_app, mocker)
    store = mocker.Mock()
    store.resolve_session_id.return_value = None
    assert cli_mod._pane_working_on(tui_app, tui_app.main, store, "ghost") is None


def test_a_store_that_throws_blocks_nobody(tui_app, mocker):
    """A guard that cannot answer must not become a refusal of its own."""
    running_pane(tui_app, mocker)
    store = mocker.Mock()
    store.resolve_session_id.side_effect = RuntimeError("database is locked")
    assert cli_mod._pane_working_on(tui_app, tui_app.main, store, "s1") is None


def test_compact_is_refused_while_that_agent_is_working(repl, mocker, cfg):
    """It rewrites the very history the turn in flight is still appending to,
    and its spinner would mark a working pane idle on the way out."""
    compact = mocker.patch.object(cli_mod.CodingAgent, "compact_history",
                                  new=mocker.AsyncMock(return_value="compacted"))
    mocker.patch.object(cli_mod, "_turn_in_flight", return_value=True)
    echo = mocker.patch.object(cli_mod, "_echo")
    repl(["/compact"])
    compact.assert_not_awaited()
    assert any("Not while" in str(c.args[0]) for c in echo.call_args_list if c.args)


def test_compact_is_not_refused_at_an_idle_prompt(repl, mocker, cfg):
    """The guard must fire only while a turn is in flight — an idle /compact
    goes on to the ordinary session check, whatever that then says."""
    mocker.patch.object(cli_mod.CodingAgent, "compact_history",
                        new=mocker.AsyncMock(return_value="compacted"))
    echo = mocker.patch.object(cli_mod, "_echo")
    repl(["/compact"])
    assert not any("Not while" in str(c.args[0]) for c in echo.call_args_list if c.args)


def test_delete_is_refused_while_a_turn_runs_on_that_session(repl, mocker, cfg):
    """Deleting the rows a running turn is writing to leaves it appending to
    a session that no longer exists."""
    mocker.patch.object(cli_mod, "_pane_working_on", return_value="audit")
    deleted = mocker.patch.object(cli_mod.SessionStore, "delete_session")
    echo = mocker.patch.object(cli_mod, "_echo")
    repl(["/delete something"])
    deleted.assert_not_called()
    assert any("Not while" in str(c.args[0]) for c in echo.call_args_list if c.args)


def test_delete_of_an_idle_session_still_works(repl, mocker, cfg):
    mocker.patch.object(cli_mod, "_pane_working_on", return_value=None)
    deleted = mocker.patch.object(cli_mod.SessionStore, "delete_session", return_value=True)
    repl(["/delete old-one"])
    deleted.assert_called_once_with("old-one")


def test_a_read_only_command_is_allowed_mid_turn(repl, mocker, cfg):
    """The point of the whole thing: /cost, /sessions and friends are safe
    while an agent works and must stay reachable."""
    mocker.patch.object(cli_mod, "_turn_in_flight", return_value=True)
    report = mocker.patch.object(cli_mod, "_report_cost")
    repl(["/cost"])
    report.assert_called_once()


# ---------------- /model with no argument ----------------


def test_bare_model_lists_what_the_server_has(repl, mocker, cfg):
    """It used to raise NameError on a `prompt_session` that no longer exists
    anywhere in the module, which took the whole REPL down — /model was
    simply unusable."""
    mocker.patch.object(cli_mod, "list_models",
                        new=mocker.AsyncMock(return_value=["alpha", "beta"]))
    echo = mocker.patch.object(cli_mod, "_echo")
    repl(["/model"])
    said = " ".join(str(c.args[0]) for c in echo.call_args_list if c.args)
    assert "alpha" in said and "beta" in said


def test_bare_model_marks_the_one_in_use(repl, mocker, cfg):
    cfg.model = "beta"
    mocker.patch.object(cli_mod, "list_models",
                        new=mocker.AsyncMock(return_value=["alpha", "beta"]))
    echo = mocker.patch.object(cli_mod, "_echo")
    repl(["/model"])
    marked = [str(c.args[0]) for c in echo.call_args_list
              if c.args and str(c.args[0]).strip().startswith("*")]
    assert marked and "beta" in marked[0]


def test_bare_model_says_how_to_switch(repl, mocker, cfg):
    mocker.patch.object(cli_mod, "list_models",
                        new=mocker.AsyncMock(return_value=["alpha"]))
    echo = mocker.patch.object(cli_mod, "_echo")
    repl(["/model"])
    said = " ".join(str(c.args[0]) for c in echo.call_args_list if c.args)
    assert "/model <name>" in said


def test_bare_model_populates_a_completion_per_model(repl, mocker, cfg):
    """The list is read and the choice is typed, so the menu has to know the
    names for that to be a usable pair."""
    mocker.patch.object(cli_mod, "list_models",
                        new=mocker.AsyncMock(return_value=["alpha", "beta"]))
    repl(["/model"])
    assert "/model alpha" in repl.captured["commands"]
    assert "/model beta" in repl.captured["commands"]


def test_bare_model_survives_a_server_with_no_models(repl, mocker, cfg):
    mocker.patch.object(cli_mod, "list_models", new=mocker.AsyncMock(return_value=[]))
    echo = mocker.patch.object(cli_mod, "_echo")
    repl(["/model"])
    said = " ".join(str(c.args[0]) for c in echo.call_args_list if c.args)
    assert "/model <name>" not in said       # nothing to switch to


def test_bare_model_reports_an_unreachable_server_without_dying(repl, mocker, cfg):
    from omni.llm_client import LLMError
    mocker.patch.object(cli_mod, "list_models",
                        new=mocker.AsyncMock(side_effect=LLMError("connection refused")))
    echo = mocker.patch.object(cli_mod, "_echo")
    repl(["/model"])
    said = " ".join(str(c.args[0]) for c in echo.call_args_list if c.args)
    assert "connection refused" in said


def test_the_module_references_no_prompt_session(repl):
    """A lint of its own: the name was read in one place and written in none,
    so nothing but running the command revealed it."""
    import re
    from pathlib import Path
    source = Path(cli_mod.__file__).read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("#"))
    assert not re.search(r"\bprompt_session\b", code)
