"""ui.py — pure formatting helpers plus smoke coverage of every renderer.

The module-level rich Console is swapped for one writing to a StringIO with
a fixed width, so output is deterministic and assertable. Renderers that
only draw (panels, tables) are checked for their key content rather than
exact layout, which would make the tests brittle against styling tweaks.
"""

import asyncio
import io

import pytest
from rich.console import Console

from omni import ui


@pytest.fixture
def cap(mocker):
    """Capture ui.console output as plain text."""
    buf = io.StringIO()
    mocker.patch.object(ui, "console", Console(file=buf, width=100, force_terminal=False,
                                               legacy_windows=False))
    return buf


def flat(buf):
    """Collapse wrapped/padded output so assertions aren't width-sensitive."""
    return " ".join(buf.getvalue().split())


# ---------------- _format_elapsed ----------------

@pytest.mark.parametrize("seconds,expected", [
    (0.04, "0.0s"), (7.77, "7.8s"), (59.94, "59.9s"),
    (60, "1m 00s"), (125, "2m 05s"), (3600, "1h 00m 00s"), (3725.9, "1h 02m 05s"),
])
def test_format_elapsed(seconds, expected):
    assert ui._format_elapsed(seconds) == expected


# ---------------- _diff_stats ----------------

DIFF = """--- a/x.py
+++ b/x.py
@@ -1,3 +1,4 @@
 keep
-gone
+added
+also added
"""


def test_diff_stats_ignores_file_headers():
    assert ui._diff_stats(DIFF) == (2, 1)


def test_diff_stats_empty():
    assert ui._diff_stats("") == (0, 0)


def test_diff_stats_all_additions():
    assert ui._diff_stats("@@ -0,0 +1,2 @@\n+a\n+b\n") == (2, 0)


# ---------------- _render_diff ----------------

def test_render_diff_numbers_and_marks_lines():
    out = ui._render_diff(DIFF).plain
    assert "-gone" in out and "+added" in out
    assert "--- a/x.py" not in out and "+++ b/x.py" not in out   # headers dropped
    assert "@@ -1,3 +1,4 @@" in out                              # hunk header kept


def test_render_diff_gutter_tracks_line_numbers():
    out = ui._render_diff("@@ -10,2 +10,2 @@\n keep\n-old\n+new\n").plain
    numbered = [l for l in out.splitlines() if l.strip()]
    assert any(l.strip().startswith("10") for l in numbered)


def test_render_diff_without_hunk_header_does_not_crash():
    assert "+lonely" in ui._render_diff("+lonely\n").plain


# ---------------- summaries ----------------

@pytest.mark.parametrize("result,expected", [
    ("(no matches)", "0 matches found"),
    ("a.py:1:x", "1 match found"),
    ("a.py:1:x\nb.py:2:y", "2 matches found"),
])
def test_search_summary_counts(result, expected):
    assert ui._search_summary(result) == expected


def test_search_summary_flags_early_stop():
    out = ui._search_summary("a:1:x\n...[stopped at 1000 matches — narrow it]")
    assert "stopped early" in out


def test_read_summary_reports_lines_and_chars():
    assert ui._read_summary("a\nb\nc") == "3 lines (5 chars)"
    assert ui._read_summary("solo") == "1 line (4 chars)"


# ---------------- tool emoji / category / arg formatting ----------------

@pytest.mark.parametrize("name,emoji", [
    ("read_file", "🔍"), ("edit_file", "✏️"), ("write_file", "📝"),
    ("run_shell", "💻"), ("git_commit", "💾"), ("save_memory", "🧠"),
    ("list_resources", "📚"), ("read_resource", "📖"),
])
def test_known_tools_get_their_emoji(name, emoji):
    assert ui._emoji_for(name) == emoji


def test_unknown_unnamespaced_tool_gets_the_default_emoji():
    assert ui._emoji_for("mystery_tool") == ui._DEFAULT_TOOL_EMOJI


def test_custom_server_tools_get_a_stable_per_server_emoji():
    """Same server → same icon, across processes (crc32, not hash())."""
    a1 = ui._emoji_for("docs__search")
    a2 = ui._emoji_for("docs__summarize")
    b = ui._emoji_for("weather__forecast")
    assert a1 == a2 and a1 in ui._SERVER_EMOJI_PALETTE
    assert b in ui._SERVER_EMOJI_PALETTE


def test_server_emoji_is_deterministic():
    assert ui._emoji_for("docs__x") == ui._emoji_for("docs__y") == ui._emoji_for("docs__z")


def test_format_args_quotes_strings_and_truncates():
    assert ui._format_args({"path": "a.py"}) == 'path="a.py"'
    assert ui._format_args({"n": 3, "flag": True}) == "n=3, flag=True"
    out = ui._format_args({"content": "x" * 200}, max_len=10)
    assert "…" in out and len(out) < 60


def test_call_str_includes_emoji_and_args():
    out = ui._call_str("read_file", {"path": "a.py"})
    assert "🔍" in out and "read_file" in out and "a.py" in out


def test_call_str_marks_malformed_args():
    assert "<malformed arguments>" in ui._call_str("read_file", None)


# ---------------- _result_line ----------------

def test_result_line_ok_and_failure_icons():
    ok, _ = ui._result_line("run_shell", {}, "exit_code: 0", True)
    bad, _ = ui._result_line("run_shell", {}, "ERROR: nope", False)
    assert "✓" in ok and "✗" in bad


def test_result_line_uses_search_and_read_summaries():
    line, _ = ui._result_line("search_files", {}, "a.py:1:x\nb.py:2:y", True)
    assert "2 matches found" in line
    line, _ = ui._result_line("read_file", {}, "l1\nl2", True)
    assert "2 lines" in line


def test_result_line_extracts_diff_body_and_stats_for_writes():
    result = "Wrote a.py.\n" + DIFF
    line, diff_body = ui._result_line("write_file", {"path": "a.py"}, result, True)
    assert "+2 -1" in line and "a.py" in line
    assert diff_body is not None and "+added" in diff_body


def test_result_line_no_diff_for_failed_write():
    line, diff_body = ui._result_line("edit_file", {"path": "a"}, "ERROR: not found", False)
    assert diff_body is None and "ERROR" in line


def test_result_line_appends_duration_when_given():
    with_time, _ = ui._result_line("read_file", {}, "x", True, duration=2.5)
    without, _ = ui._result_line("read_file", {}, "x", True)
    assert "2.5s" in with_time and "s" not in without.replace("chars", "")


def test_result_line_handles_empty_result_and_non_dict_args():
    line, _ = ui._result_line("odd_tool", None, "", False)
    assert "✗" in line


# ---------------- step_display ----------------

def test_step_display_renders_call_and_result(cap):
    ui.step_display([{"name": "read_file", "args": {"path": "a.py"},
                      "result": "l1\nl2", "ok": True, "duration": 0.5}])
    out = flat(cap)
    assert "read_file" in out and "a.py" in out and "2 lines" in out and "⎿" in out


def test_step_display_renders_every_parallel_call(cap):
    ui.step_display([
        {"name": "read_file", "args": {"path": "a"}, "result": "x", "ok": True},
        {"name": "list_dir", "args": {"path": "."}, "result": "y", "ok": True},
    ])
    out = flat(cap)
    assert "read_file" in out and "list_dir" in out


def test_step_display_includes_the_diff_for_writes(cap):
    ui.step_display([{"name": "write_file", "args": {"path": "a.py"},
                      "result": "Wrote a.py.\n" + DIFF, "ok": True}])
    assert "+added" in flat(cap)


def test_step_display_handles_malformed_args(cap):
    ui.step_display([{"name": "read_file", "args": None,
                      "result": "ERROR: malformed arguments", "ok": False}])
    assert "malformed" in flat(cap)


# ---------------- SlashCommandCompleter ----------------

def complete(commands, text):
    from prompt_toolkit.document import Document
    c = ui.SlashCommandCompleter(commands)
    return [x.text for x in c.get_completions(Document(text), None)]


def test_completer_only_fires_on_slash():
    cmds = {"/exit": "leave", "/model": "switch"}
    assert complete(cmds, "") == []
    assert complete(cmds, "hello") == []
    assert set(complete(cmds, "/")) == {"/exit", "/model"}


def test_completer_filters_by_prefix():
    cmds = {"/exit": "", "/model": "", "/mcp": ""}
    assert complete(cmds, "/m") == ["/model", "/mcp"]
    assert complete(cmds, "/exi") == ["/exit"]
    assert complete(cmds, "/zzz") == []


def test_completer_reads_the_live_dict():
    """cli.py mutates the same dict after the session is built."""
    cmds = {"/exit": ""}
    c = ui.SlashCommandCompleter(cmds)
    cmds["/added-later"] = "new"
    from prompt_toolkit.document import Document
    assert "/added-later" in [x.text for x in c.get_completions(Document("/ad"), None)]


def test_completer_exposes_descriptions_as_meta():
    from prompt_toolkit.document import Document
    c = ui.SlashCommandCompleter({"/model": "switch the model"})
    (item,) = list(c.get_completions(Document("/mo"), None))
    assert "switch the model" in str(item.display_meta)


# ---------------- _TickingSpinner / thinking ----------------

async def test_ticking_spinner_ticks_an_elapsed_suffix(cap):
    spinner = ui.thinking("Working")
    with spinner:
        await asyncio.sleep(0.35)               # let the ticker fire a few times
        shown = spinner._spinner.text
    assert "Working" in str(shown) and "s)" in str(shown)


async def test_ticking_spinner_update_replaces_the_label(cap):
    spinner = ui.thinking("Thinking")
    with spinner:
        spinner.update("[bold yellow]Retrying 1/3[/bold yellow]")
        await asyncio.sleep(0.35)
        shown = str(spinner._spinner.text)
    assert "Retrying 1/3" in shown


async def test_ticking_spinner_cancels_its_task_on_exit(cap):
    spinner = ui.thinking("x")
    with spinner:
        await asyncio.sleep(0.05)
    await asyncio.sleep(0)                      # let the cancellation settle
    assert spinner._task.done()


async def test_ticking_spinner_survives_a_ticker_error(cap, mocker):
    """The ticker is cosmetic — a failure in it must never surface over the
    real work being awaited."""
    spinner = ui.thinking("x")
    # Patch what _tick itself calls, not Live.__exit__'s final refresh.
    mocker.patch.object(spinner._spinner, "update", side_effect=RuntimeError("render boom"))
    with spinner:
        await asyncio.sleep(0.25)               # no exception escapes here
    assert spinner._task.done()


# ---------------- renderers (smoke + key content) ----------------

def test_banner_and_header(cap):
    ui.banner("my task", "my-model")
    ui.header("my-model", "my-session")
    out = flat(cap)
    assert "my task" in out and "my-model" in out and "my-session" in out


def test_intent_panel(cap):
    from omni.intent import Intent
    ui.intent_panel(Intent(task_type="bugfix", summary="fix it", target_files=["a.py"],
                           constraints=["be quick"], risk_level="high"),
                    {"a.py": True})
    out = flat(cap)
    assert "bugfix" in out and "fix it" in out and "high" in out and "a.py" in out


def test_intent_panel_flags_low_confidence(cap):
    from omni.intent import Intent
    ui.intent_panel(Intent(confident=False), {})
    assert "low confidence" in flat(cap)


def test_high_risk_warning(cap):
    ui.high_risk_warning()
    assert "High-risk" in flat(cap) and "auto-approve" in flat(cap)


def test_final_result_renders_markdown(cap):
    ui.final_result("# Heading\n\nBody text")
    out = flat(cap)
    assert "Heading" in out and "Body text" in out


def test_history_panel_shows_each_role(cap):
    ui.history_panel([
        {"role": "system", "content": "hidden"},
        {"role": "user", "content": "my question"},
        {"role": "assistant", "content": "my answer"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "content": "tool result"},
    ])
    out = flat(cap)
    assert "my question" in out and "my answer" in out and "read_file" in out
    assert "hidden" not in out          # system messages aren't replayed


def test_sessions_table(cap):
    ui.sessions_table([{"id": "abc12345", "name": "nm", "status": "done",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                        "model": "qwen", "task": "the task"}])
    out = flat(cap)
    assert "abc12345" in out and "done" in out and "the task" in out


def test_sessions_table_truncates_long_tasks(cap):
    ui.sessions_table([{"id": "i", "name": None, "status": "running",
                        "updated_at": "t", "model": "m", "task": "y" * 200}])
    assert "…" in cap.getvalue()


@pytest.mark.parametrize("status", ["done", "running", "error", "max_steps", "interrupted"])
def test_sessions_table_renders_each_status(cap, status):
    ui.sessions_table([{"id": "i", "name": None, "status": status,
                        "updated_at": "t", "model": "m", "task": "t"}])
    assert status in flat(cap)


def test_resources_table(cap):
    ui.resources_table({
        "file:///a.md": {"server": "docs", "name": "std", "description": "Standards",
                         "mime_type": "text/markdown", "size": 1, "template": False,
                         "shadowed_by": []},
        "file:///{d}.log": {"server": "docs", "name": "log", "description": "Daily",
                            "mime_type": "", "size": None, "template": True,
                            "shadowed_by": []},
    })
    out = flat(cap)
    assert "file:///a.md" in out and "Standards" in out and "template" in out


def test_resources_table_notes_shadowing(cap):
    ui.resources_table({"x://1": {"server": "a", "name": "", "description": "d",
                                  "mime_type": "", "size": None, "template": False,
                                  "shadowed_by": ["b"]}})
    assert "also on" in flat(cap)


def test_resources_table_empty(cap):
    ui.resources_table({})
    assert "No resources" in flat(cap)


def test_resource_content(cap):
    ui.resource_content("file:///a.md", "the body")
    out = flat(cap)
    assert "file:///a.md" in out and "the body" in out


def test_resource_content_empty(cap):
    ui.resource_content("x://1", "")
    assert "(empty)" in flat(cap)


def test_mcp_status_shows_connected_and_failed(cap):
    ui.mcp_status([
        {"name": "built-in", "connected": True, "connected_for": 12.0, "error": None,
         "deferred": False, "tool_count": 18, "target": "python -m omni.mcp_server"},
        {"name": "docs", "connected": False, "connected_for": None, "error": "boom",
         "deferred": True, "tool_count": 0, "target": "https://h/mcp"},
    ])
    out = flat(cap)
    assert "built-in" in out and "18" in out
    assert "docs" in out and "boom" in out


def test_interrupted_and_compacted_and_btw(cap):
    ui.interrupted()
    ui.compacted(12, 340, 1.5)
    ui.btw_answer("what is x?", "x is y")
    out = flat(cap)
    assert "Interrupted" in out
    assert "12" in out and "340" in out and "1.5s" in out
    assert "what is x?" in out and "x is y" in out


def test_warning_and_error(cap):
    ui.warning("careful")
    ui.error("broken")
    out = flat(cap)
    assert "careful" in out and "broken" in out


def test_elapsed_note(cap):
    ui.elapsed_note("Responded", 3.25)
    assert "Responded" in flat(cap) and "3.2s" in flat(cap)


# ---------------- request_approval ----------------

async def test_request_approval_shows_edit_diff_and_asks(cap, mocker):
    client = mocker.AsyncMock()
    client.preview_edit.return_value = (True, DIFF)
    mocker.patch.object(ui.Confirm, "ask", return_value=True)
    assert await ui.request_approval("edit_file", {"path": "a.py", "old_str": "x", "new_str": "y"}, client)
    out = flat(cap)
    assert "edit_file" in out and "a.py" in out and "+2" in out and "-1" in out


async def test_request_approval_rejects_when_preview_fails(cap, mocker):
    client = mocker.AsyncMock()
    client.preview_edit.return_value = (False, "old_str not found")
    confirm = mocker.patch.object(ui.Confirm, "ask", return_value=True)
    assert await ui.request_approval("edit_file", {"path": "a.py"}, client) is False
    confirm.assert_not_called()          # never asks when the edit can't apply
    assert "not found" in flat(cap)


@pytest.mark.parametrize("is_new,label", [(True, "new"), (False, "overwrite")])
async def test_request_approval_write_file_labels_new_vs_overwrite(cap, mocker, is_new, label):
    client = mocker.AsyncMock()
    client.preview_write.return_value = (is_new, DIFF)
    mocker.patch.object(ui.Confirm, "ask", return_value=True)
    await ui.request_approval("write_file", {"path": "a.py", "content": "x"}, client)
    assert label in flat(cap)


async def test_request_approval_shows_shell_command(cap, mocker):
    mocker.patch.object(ui.Confirm, "ask", return_value=False)
    assert await ui.request_approval("run_shell", {"command": "rm -i x"}, mocker.AsyncMock()) is False
    assert "rm -i x" in flat(cap)


async def test_request_approval_falls_back_to_json_for_other_tools(cap, mocker):
    mocker.patch.object(ui.Confirm, "ask", return_value=True)
    await ui.request_approval("docs__publish", {"target": "prod"}, mocker.AsyncMock())
    out = flat(cap)
    assert "docs__publish" in out and "prod" in out


# ---------------- prompt_task_async ----------------

async def test_prompt_task_async_reads_from_the_session(cap, mocker):
    session = mocker.Mock()
    session.prompt_async = mocker.AsyncMock(return_value="  my typed task  ")
    assert await ui.prompt_task_async(session) == "  my typed task  "
    session.prompt_async.assert_awaited_once()
