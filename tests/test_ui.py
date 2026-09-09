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


@pytest.fixture(autouse=True)
def no_tui():
    """ui keeps the full-screen app in module state; a test elsewhere that ran
    the REPL would otherwise leave one registered and silently reroute every
    assertion here into its transcript."""
    ui.use_tui(None)
    yield
    ui.use_tui(None)


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
    ("read_file", "🔍"), ("edit_file", "📝"), ("write_file", "📄"),
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
    assert ui._format_args({"path": "a.py"}) == ('path="a.py"', False)
    assert ui._format_args({"n": 3, "flag": True}) == ("n=3, flag=True", False)
    out, truncated = ui._format_args({"content": "x" * 200}, max_len=10)
    assert "…" in out and len(out) < 60 and truncated is True


def test_format_args_budget_follows_the_terminal_width(mocker):
    """A fixed 60-character cap clipped arguments that a wide terminal had
    room for; only genuinely long values should be cut now."""
    value = "y" * 300
    mocker.patch.object(ui, "console", Console(width=400, force_terminal=False))
    wide, wide_cut = ui._format_args({"content": value})
    mocker.patch.object(ui, "console", Console(width=80, force_terminal=False))
    narrow, narrow_cut = ui._format_args({"content": value})
    assert len(wide) > len(narrow)
    assert wide_cut is False and narrow_cut is True


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
    real work being awaited, and must not stop the ticker either: catching
    around the loop instead of inside it froze the glyph and the elapsed
    counter for the rest of the turn after one bad frame."""
    spinner = ui.thinking("x")
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("render boom")

    # Patch what _tick itself calls, not Live.__exit__'s final refresh.
    mocker.patch.object(spinner._spinner, "update", side_effect=boom)
    with spinner:
        await asyncio.sleep(0.5)                # no exception escapes here
        assert not spinner._task.done()         # still ticking
    assert calls["n"] > 1                       # kept trying, frame after frame
    await asyncio.sleep(0)                      # let the cancellation land
    assert spinner._task.done()


# ---------------- renderers (smoke + key content) ----------------

def test_banner_and_header(cap):
    ui.banner("my task", "my-model")
    ui.header("my-session", "/tmp/proj")
    out = flat(cap)
    assert "my task" in out and "my-model" in out       # banner (one-shot runs)
    assert "my-session" in out and "Omni Coder" in out  # header


def test_header_does_not_name_a_model(cap):
    """It's scrollback the moment it prints, so any model named here goes
    stale on the first /model switch."""
    ui.header("my-session", "/tmp/proj")
    assert "qwen" not in flat(cap).lower()


# ---------------- reasoning ----------------

LONG_REASONING = ("First I check the frame. " * 20) + "\n\nThen I summarise."


def test_reasoning_note_is_collapsed(cap):
    ui.reasoning_note(LONG_REASONING)
    out = cap.getvalue()
    assert "▸ reasoning" in out and "/reasoning to expand" in out
    assert "chars" in out
    body = [l for l in out.split("\n") if "│" in l]
    assert len(body) == 2 and out.rstrip().endswith("…")   # clipped, with a marker


def test_reasoning_full_shows_everything(cap):
    ui.reasoning_full(LONG_REASONING)
    out = cap.getvalue()
    assert "▾ reasoning" in out                            # disclosure flipped open
    assert "Then I summarise." in out
    assert not out.rstrip().endswith("…")


def test_reasoning_gutter_wraps_to_the_terminal(mocker):
    buf = io.StringIO()
    mocker.patch.object(ui, "console", Console(file=buf, width=50, force_terminal=False,
                                               legacy_windows=False))
    ui.reasoning_full(LONG_REASONING)
    assert all(len(l) <= 50 for l in buf.getvalue().split("\n"))


def test_final_result_calls_out_an_empty_answer(cap):
    """A blank panel is indistinguishable from "the terminal isn't rendering
    the response", which is the exact confusion an empty reply causes."""
    ui.final_result("")
    out = flat(cap)
    assert "empty response" in out and "agent_run.log" in out


def test_final_result_renders_real_text(cap):
    ui.final_result("**done** — created `note.txt`")
    assert "note.txt" in flat(cap)


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


def tool_entry(name, description="", **flags):
    base = {"name": name, "real_name": name.split("__")[-1], "description": description,
            "deferred": False, "revealed": False, "internal": False}
    base.update(flags)
    return base


def test_server_tools_table_lists_tools_with_emoji_and_description(cap):
    ui.server_tools_table("built-in", [
        tool_entry("read_file", "Read a file, optionally a line range."),
        tool_entry("write_file", "Create a NEW file with content."),
    ])
    out = flat(cap)
    assert "read_file" in out and "Read a file" in out
    assert "🔍" in out and "📄" in out
    assert "2 of 2 callable" in out


def test_server_tools_table_omits_status_column_when_nothing_notable(cap):
    ui.server_tools_table("docs", [tool_entry("docs__a", "does a")])
    assert "status" not in flat(cap)


def test_server_tools_table_shows_deferred_and_revealed(cap):
    ui.server_tools_table("docs", [
        tool_entry("docs__a", "hidden one", deferred=True),
        tool_entry("docs__b", "surfaced one", revealed=True),
    ])
    out = flat(cap)
    assert "deferred" in out and "revealed" in out
    assert "1 of 2 callable" in out
    assert "search_tools" in out          # explains how deferred ones load


def test_server_tools_table_flags_internal_tools(cap):
    ui.server_tools_table("built-in", [
        tool_entry("read_file", "public"),
        tool_entry("_preview_edit", "internal helper", internal=True),
    ])
    out = flat(cap)
    assert "internal" in out and "1 of 2 callable" in out


def test_server_tools_table_uses_only_the_first_description_line(cap):
    ui.server_tools_table("d", [tool_entry("d__a", "first line\nsecond line")])
    out = flat(cap)
    assert "first line" in out and "second line" not in out


def test_server_tools_table_empty(cap):
    ui.server_tools_table("docs", [])
    assert "no tools" in flat(cap)


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


# ---------------- ask_user ----------------

async def test_ask_user_numbered_choice_maps_to_the_option(cap, mocker):
    mocker.patch.object(ui.console, "input", return_value="2")
    answer = await ui.ask_user("Which store?", ["SQLite", "Postgres", "Neither"])
    assert answer == "Postgres"
    out = flat(cap)
    assert "Which store?" in out and "1. SQLite" in out and "3. Neither" in out


async def test_ask_user_takes_free_text_verbatim(cap, mocker):
    """The interesting answer is often none of the offered ones."""
    mocker.patch.object(ui.console, "input", return_value="use DuckDB instead")
    assert await ui.ask_user("Which store?", ["SQLite", "Postgres"]) == "use DuckDB instead"


async def test_ask_user_without_options_is_just_a_question(cap, mocker):
    mocker.patch.object(ui.console, "input", return_value="the second one")
    assert await ui.ask_user("Which did you mean?") == "the second one"
    assert "type your answer" in flat(cap)


@pytest.mark.parametrize("out_of_range", ["0", "4", "12"])
async def test_a_number_outside_the_options_is_treated_as_text(cap, mocker, out_of_range):
    mocker.patch.object(ui.console, "input", return_value=out_of_range)
    assert await ui.ask_user("q", ["a", "b", "c"]) == out_of_range


@pytest.mark.parametrize("boom", [EOFError(), KeyboardInterrupt()])
async def test_dismissing_the_question_returns_none(cap, mocker, boom):
    mocker.patch.object(ui.console, "input", side_effect=boom)
    assert await ui.ask_user("q", ["a"]) is None


async def test_an_empty_answer_counts_as_dismissed(cap, mocker):
    mocker.patch.object(ui.console, "input", return_value="   ")
    assert await ui.ask_user("q") is None


# ---------------- theme colour ----------------

def test_the_accent_can_be_changed(cap):
    """Renderers build their styles from ACCENT at call time, so recolouring
    is a matter of one variable — except the prompt_toolkit style map, which
    bakes it in and has to be rebuilt."""
    try:
        before = ui._PROMPT_STYLE
        ui.set_accent("#00b4d8")
        assert ui.ACCENT == "#00b4d8"
        assert ui._PROMPT_STYLE is not before
    finally:
        ui.set_accent("")
    assert ui.ACCENT == ui.DEFAULT_ACCENT


def test_a_changed_accent_reaches_the_output(mocker):
    def render():
        buf = io.StringIO()
        mocker.patch.object(ui, "console", Console(file=buf, width=70, force_terminal=True,
                                                   color_system="truecolor"))
        ui.step_display([{"index": 1, "name": "read_file", "args": {"path": "a.py"},
                          "result": "x", "ok": True, "duration": 0.1}])
        return buf.getvalue()

    try:
        assert "217;119;87" in render()          # the built-in accent
        ui.set_accent("#00b4d8")
        recoloured = render()
        assert "0;180;216" in recoloured and "217;119;87" not in recoloured
    finally:
        ui.set_accent("")


def test_an_empty_theme_colour_restores_the_default():
    ui.set_accent("#123456")
    ui.set_accent("")
    assert ui.ACCENT == ui.DEFAULT_ACCENT


# ---------------- terminal title ----------------

def test_terminal_title_uses_the_osc_escape(cap, mocker):
    """Works on xterm-family terminals, Terminal.app/iTerm and Windows
    Terminal alike — Rich emits it only when stdout is a real terminal."""
    setter = mocker.patch.object(ui.console, "set_window_title")
    ui.set_terminal_title("my-session")
    setter.assert_called_once_with("my-session")


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_title_is_ignored(cap, mocker, blank):
    setter = mocker.patch.object(ui.console, "set_window_title")
    ui.set_terminal_title(blank)
    setter.assert_not_called()


def test_windows_also_goes_through_the_console_api(cap, mocker):
    """Older Windows consoles ignore the escape and only answer to
    SetConsoleTitleW."""
    win = mocker.patch.object(ui, "_set_windows_console_title")
    mocker.patch.object(ui, "_IS_WINDOWS", True)
    mocker.patch.object(ui.console, "set_window_title")
    ui.set_terminal_title("my-session")
    win.assert_called_once_with("my-session")


def test_other_platforms_skip_the_windows_call(cap, mocker):
    win = mocker.patch.object(ui, "_set_windows_console_title")
    mocker.patch.object(ui, "_IS_WINDOWS", False)
    mocker.patch.object(ui.console, "set_window_title")
    ui.set_terminal_title("my-session")
    win.assert_not_called()


def test_a_failing_terminal_never_breaks_the_run(cap, mocker):
    mocker.patch.object(ui.console, "set_window_title", side_effect=RuntimeError("no tty"))
    ui.set_terminal_title("my-session")   # must not raise


# ---------------- resize tolerance ----------------

@pytest.mark.parametrize("width", [60, 100, 160, 220])
def test_nothing_boxed_spans_a_wide_terminal(mocker, width):
    """Scrollback belongs to the terminal: a box drawn the full width of a
    wide window is rewrapped into broken box-drawing the moment the window
    narrows, and nothing can redraw it. Capped boxes only suffer below their
    own width."""
    buf = io.StringIO()
    mocker.patch.object(ui, "console", Console(file=buf, width=width, force_terminal=False,
                                               legacy_windows=False))
    ui.header("sess", "/tmp/p")
    ui.intent_panel(_intent(), {})
    longest = max((len(l) for l in buf.getvalue().split("\n")), default=0)
    assert longest <= min(width, 100)


def _intent():
    from omni.intent import Intent
    return Intent(task_type="feature", summary="s", target_files=[], constraints=[])


# ---------------- the bottom frame ----------------

async def test_thinking_inside_a_turn_frame_supports_update(cap):
    """The retry path does `spinner = thinking(); ...; spinner.update(...)` on
    the object itself, so whatever thinking() returns has to carry .update —
    a bare @contextmanager generator raised AttributeError there."""
    async with ui.turn_frame("my-session"):
        spinner = ui.thinking("Thinking…")
        assert hasattr(spinner, "update")
        with spinner as handle:
            handle.update("[bold yellow]Thinking… (retry 1/3)[/bold yellow]")
            assert "retry 1/3" in ui._frame._label
            spinner.update("relabelled directly")
            assert "relabelled directly" in ui._frame._label


async def test_turn_frame_restores_the_label_after_a_phase(cap):
    async with ui.turn_frame("s"):
        before = ui._frame._label
        with ui.thinking("Running read_file…"):
            assert "read_file" in ui._frame._label
        assert ui._frame._label == before


async def test_turn_frame_opens_and_closes(cap):
    assert ui._frame.is_open is False
    async with ui.turn_frame("s"):
        assert ui._frame.is_open is True
    assert ui._frame.is_open is False


async def test_thinking_outside_a_frame_is_a_standalone_spinner(cap):
    spinner = ui.thinking("Parsing intent…")
    assert isinstance(spinner, ui._TickingSpinner)
    with spinner:
        spinner.update("still going")


async def test_frame_render_carries_session_chip_and_hint(cap):
    async with ui.turn_frame("probe-demo"):
        ui.console.print(ui._frame._render())
    out = flat(cap)
    assert "probe-demo" in out and "ctrl+c interrupts" in out


@pytest.mark.parametrize("width", [40, 60, 100, 200])
async def test_frame_render_is_always_five_rows(mocker, width):
    """Live repaints by stepping back over the previous render, so a frame
    line that wraps at one width and not another strands the old copy on
    screen — that was the staircase of rules seen when widening a terminal
    mid-turn."""
    buf = io.StringIO()
    mocker.patch.object(ui, "console", Console(file=buf, width=width, force_terminal=False,
                                               legacy_windows=False))
    async with ui.turn_frame("a-fairly-long-session-name", "some-long-model-name:35b"):
        ui.console.print(ui._frame._render())
    # blank, status, blank, rule, hint
    assert len(buf.getvalue().rstrip("\n").split("\n")) == 5


async def test_frame_pause_lets_the_terminal_go(cap):
    async with ui.turn_frame("s"):
        with ui._frame.pause():
            assert ui._frame._paused is True
        assert ui._frame._paused is False


def test_frame_pause_outside_a_turn_is_a_no_op():
    with ui._frame.pause():
        pass


# ---------------- the bottom frame ----------------

async def test_thinking_inside_a_turn_frame_supports_update(cap):
    """The retry path does `spinner = thinking(); ...; spinner.update(...)` on
    the object itself, so whatever thinking() returns has to carry .update —
    a bare @contextmanager generator raised AttributeError there."""
    async with ui.turn_frame("my-session"):
        spinner = ui.thinking("Thinking…")
        assert hasattr(spinner, "update")
        with spinner as handle:
            handle.update("[bold yellow]Thinking… (retry 1/3)[/bold yellow]")
            assert "retry 1/3" in ui._frame._label
            spinner.update("relabelled directly")
            assert "relabelled directly" in ui._frame._label


async def test_turn_frame_restores_the_label_after_a_phase(cap):
    async with ui.turn_frame("s"):
        before = ui._frame._label
        with ui.thinking("Running read_file…"):
            assert "read_file" in ui._frame._label
        assert ui._frame._label == before


async def test_turn_frame_opens_and_closes(cap):
    assert ui._frame.is_open is False
    async with ui.turn_frame("s"):
        assert ui._frame.is_open is True
    assert ui._frame.is_open is False


async def test_thinking_outside_a_frame_is_a_standalone_spinner(cap):
    spinner = ui.thinking("Parsing intent…")
    assert isinstance(spinner, ui._TickingSpinner)
    with spinner:
        spinner.update("still going")


async def test_frame_render_carries_session_chip_and_hint(cap):
    async with ui.turn_frame("probe-demo"):
        ui.console.print(ui._frame._render())
    out = flat(cap)
    assert "probe-demo" in out and "ctrl+c interrupts" in out


@pytest.mark.parametrize("width", [40, 60, 100, 200])
async def test_frame_render_is_always_five_rows(mocker, width):
    """Live repaints by stepping back over the previous render, so a frame
    line that wraps at one width and not another strands the old copy on
    screen — that was the staircase of rules seen when widening a terminal
    mid-turn."""
    buf = io.StringIO()
    mocker.patch.object(ui, "console", Console(file=buf, width=width, force_terminal=False,
                                               legacy_windows=False))
    async with ui.turn_frame("a-fairly-long-session-name", "some-long-model-name:35b"):
        ui.console.print(ui._frame._render())
    # blank, status, blank, rule, hint
    assert len(buf.getvalue().rstrip("\n").split("\n")) == 5


async def test_frame_pause_lets_the_terminal_go(cap):
    async with ui.turn_frame("s"):
        with ui._frame.pause():
            assert ui._frame._paused is True
        assert ui._frame._paused is False


def test_frame_pause_outside_a_turn_is_a_no_op():
    with ui._frame.pause():
        pass


# ---------------- prompt_task_async ----------------
