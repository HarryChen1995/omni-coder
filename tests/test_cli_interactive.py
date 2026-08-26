"""The interactive REPL loop (_interactive).

Driven by scripting `_read_task` to return a sequence of typed lines ending
in /exit, with the MCP client and CodingAgent.run mocked. That exercises
every slash command's branch plus the turn plumbing — session id carry-over,
header refresh, cancellation — without a terminal or a model.
"""

import asyncio
import json

import pytest

from omni import cli as cli_mod


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


@pytest.fixture
def repl(mocker, client, cfg):
    """Run the REPL over a scripted list of inputs. Returns a helper that
    yields the mocks for assertions."""
    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch("omni.ui.final_result")
    mocker.patch("omni.ui.interrupted")
    mocker.patch("omni.ui.thinking")

    def run(inputs, run_result="the answer", run_side_effect=None, **kwargs):
        script = list(inputs) + ["/exit"]
        mocker.patch.object(cli_mod, "_read_task",
                            mocker.AsyncMock(side_effect=script))
        agent_run = mocker.patch.object(
            cli_mod.CodingAgent, "run",
            mocker.AsyncMock(return_value=run_result, side_effect=run_side_effect))
        # prompt_toolkit's PromptSession needs a real terminal; the REPL only
        # uses the object as an opaque handle here since _read_task is mocked.
        mocker.patch.object(cli_mod, "_interactive_prompt_session", create=True)
        mocker.patch("prompt_toolkit.PromptSession", mocker.Mock())
        mocker.patch("prompt_toolkit.patch_stdout.patch_stdout", mocker.MagicMock())
        asyncio.run(cli_mod._interactive(cfg, kwargs.get("resume"), kwargs.get("session_name")))
        return agent_run

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
    mocker.patch.object(cli_mod, "_read_task", mocker.AsyncMock(side_effect=[quit_cmd, "never runs"]))
    agent_run = mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock())
    repl([quit_cmd])
    agent_run.assert_not_awaited()


def test_session_id_is_carried_between_turns(mocker, client, cfg):
    """Turn 2 must resume the session turn 1 created, not start a fresh one."""
    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch("omni.ui.final_result")
    mocker.patch.object(cli_mod, "_read_task",
                        mocker.AsyncMock(side_effect=["first", "second", "/exit"]))
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
    mocker.patch.object(cli_mod, "_read_task", mocker.AsyncMock(side_effect=EOFError))
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
    mocker.patch.object(cli_mod, "_read_task", mocker.AsyncMock(side_effect=["/compact", "/exit"]))
    compact = mocker.patch.object(cli_mod.CodingAgent, "compact_history",
                                 mocker.AsyncMock(return_value="Compacted 30 down to 5"))
    mocker.patch("prompt_toolkit.PromptSession", mocker.Mock())
    mocker.patch("prompt_toolkit.patch_stdout.patch_stdout", mocker.MagicMock())
    asyncio.run(cli_mod._interactive(cfg, "my-session", None))
    assert compact.await_args.args[0] == sid          # real id, not the name


# ---------------- /model ----------------

def test_model_command_lists_models(mocker, client, cfg, capsys):
    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch.object(cli_mod, "list_models", mocker.AsyncMock(return_value=["a", "b"]))
    mocker.patch.object(cli_mod, "_read_task", mocker.AsyncMock(side_effect=["/model", "/exit"]))
    mocker.patch("prompt_toolkit.PromptSession", mocker.Mock())
    mocker.patch("prompt_toolkit.patch_stdout.patch_stdout", mocker.MagicMock())
    # No usable picker without a real terminal -> falls back to the printed list.
    mocker.patch("prompt_toolkit.shortcuts.radiolist_dialog", side_effect=Exception("no tty"))
    with pytest.raises(Exception):
        asyncio.run(cli_mod._interactive(cfg, None, None))


def test_model_name_command_switches_without_redrawing_the_header(repl, mocker, cfg, capsys):
    """The header is startup furniture: a /model switch echoes the new name
    instead of dropping a second copy of the box into the transcript."""
    header = mocker.patch.object(cli_mod, "_print_header")
    repl(["/model llama3.1:latest"])
    assert cfg.model == "llama3.1:latest"
    assert "llama3.1:latest" in capsys.readouterr().out
    assert header.call_count == 1          # the one at startup


def test_header_is_not_redrawn_when_the_session_gets_its_id(repl, mocker):
    header = mocker.patch.object(cli_mod, "_print_header")
    mocker.patch.object(cli_mod.CodingAgent, "session_id", "abc12345", create=True)
    repl(["first task", "second task"])
    assert header.call_count == 1


def test_model_list_failure_is_reported(mocker, client, cfg, capsys):
    from omni.llm_client import LLMError
    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch.object(cli_mod, "list_models", mocker.AsyncMock(side_effect=LLMError("no /v1/models")))
    mocker.patch.object(cli_mod, "_read_task", mocker.AsyncMock(side_effect=["/model", "/exit"]))
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
    captured = {}
    mocker.patch("omni.ui.SlashCommandCompleter",
                 side_effect=lambda commands: captured.setdefault("commands", commands))
    repl([])
    assert "/mcp tools built-in" in captured["commands"]
    assert "/mcp tools docs" in captured["commands"]


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
    captured = {}
    mocker.patch("omni.ui.SlashCommandCompleter",
                 side_effect=lambda commands: captured.setdefault("commands", commands))
    repl([])
    entry = captured["commands"]["/docs:summarize "]
    assert "Summarize" in entry and "<path>" in entry and "[style]" in entry


def test_unknown_slash_command_is_passed_through_as_a_task(repl, client):
    """Not every "/..." line is a command — an unmatched one is still a task."""
    client.list_prompts.return_value = {}
    agent_run = repl(["/not-a-command at all"])
    assert agent_run.await_args.args[0] == "/not-a-command at all"


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
    mocker.patch.object(cli_mod, "_read_task", mocker.AsyncMock(side_effect=["/exit"]))
    mocker.patch("prompt_toolkit.PromptSession", mocker.Mock())
    mocker.patch("prompt_toolkit.patch_stdout.patch_stdout", mocker.MagicMock())
    asyncio.run(cli_mod._interactive(cfg, "some-session", None))
    shown.assert_called_once()


def test_model_completions_are_registered_when_available(repl, client, mocker):
    mocker.patch.object(cli_mod, "list_models", mocker.AsyncMock(return_value=["m1", "m2"]))
    captured = {}
    mocker.patch("omni.ui.SlashCommandCompleter",
                 side_effect=lambda commands: captured.setdefault("commands", commands))
    repl([])
    assert "/model m1" in captured["commands"] and "/model m2" in captured["commands"]


def test_missing_v1_models_endpoint_is_tolerated(repl, client, mocker):
    from omni.llm_client import LLMError
    mocker.patch.object(cli_mod, "list_models", mocker.AsyncMock(side_effect=LLMError("404")))
    repl(["a task"])            # startup must not fail
