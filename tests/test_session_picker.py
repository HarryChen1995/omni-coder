"""The list bare `--resume` opens."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from omni import ui
from omni.session_picker import (
    RESERVED_KEYS, SessionPicker, relative_age, scroll_window, session_detail,
    session_title, visible_sessions,
)

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


def session(**overrides):
    base = {
        "id": "abc123", "name": None, "task": "fix the importer",
        "model": "qwen3-coder", "status": "done", "messages": 8,
        "project_root": "/work/omni", "summary": "", "branch": "master",
        "updated_at": (NOW - timedelta(days=2)).isoformat(),
        "created_at": (NOW - timedelta(days=2)).isoformat(),
    }
    base.update(overrides)
    return base


def styles(fragments):
    return [style for style, _ in fragments]


def text(fragments):
    return "".join(body for _, body in fragments)


# ---- the row text ----

@pytest.mark.parametrize("delta, expected", [
    (timedelta(seconds=3), "just now"),
    (timedelta(seconds=90), "1 minute ago"),
    (timedelta(minutes=45), "45 minutes ago"),
    (timedelta(hours=5), "5 hours ago"),
    (timedelta(days=1), "1 day ago"),
    (timedelta(days=3), "3 days ago"),
    (timedelta(days=12), "1 week ago"),
    (timedelta(days=200), "6 months ago"),
])
def test_relative_age(delta, expected):
    assert relative_age((NOW - delta).isoformat(), NOW) == expected


def test_relative_age_keeps_an_unparseable_timestamp():
    """Better a raw string than an empty column — it is the row's only clue
    about when the session ran."""
    assert relative_age("not a date", NOW) == "not a date"


def test_the_title_is_the_session_name_when_it_has_one():
    assert session_title(session(name="auth rewrite")) == "auth rewrite"


def test_the_title_falls_back_to_the_task_not_the_id():
    """The id is a hex blob nobody recognises; it belongs in the detail
    line, not the headline."""
    assert session_title(session(name=None)) == "fix the importer"


def test_a_long_task_is_truncated_to_one_line():
    title = session_title(session(name=None, task="word " * 40))
    assert len(title) == 72 and title.endswith("…")


def test_the_title_falls_back_to_the_id_when_there_is_nothing_else():
    assert session_title(session(name=None, task="")) == "abc123"


def test_the_detail_line_carries_age_model_size_and_id():
    detail = session_detail(session(), NOW)
    assert "2 days ago" in detail and "qwen3-coder" in detail
    assert "8 messages" in detail and "abc123" in detail


def test_the_detail_line_says_message_not_messages_for_one():
    assert "1 message  " in session_detail(session(messages=1), NOW) + "  "


def test_a_session_with_no_messages_yet_shows_no_count():
    assert "0 message" not in session_detail(session(messages=0), NOW)


# ---- filtering ----

def test_only_this_projects_sessions_are_listed():
    rows = visible_sessions([session(id="mine"), session(id="theirs", project_root="/work/other")],
                             project_root="/work/omni")
    assert [r["id"] for r in rows] == ["mine"]


def test_no_project_root_means_every_project():
    rows = visible_sessions([session(id="mine"), session(id="theirs", project_root="/work/other")])
    assert len(rows) == 2


def test_only_this_branchs_sessions_are_listed():
    rows = visible_sessions([session(id="here"), session(id="elsewhere", branch="feature/x")],
                             branch="master")
    assert [r["id"] for r in rows] == ["here"]


def test_a_session_with_no_branch_recorded_survives_the_branch_filter():
    """Sessions from before the column existed, and work done outside a
    checkout, belong to no branch in particular — hiding them would make
    them unreachable from the list entirely."""
    rows = visible_sessions([session(id="old", branch=None), session(id="new")], branch="master")
    assert [r["id"] for r in rows] == ["old", "new"]


def test_the_search_matches_words_in_any_order():
    """"importer fix" has to find "fix the importer" — a substring match
    would not, and word order is exactly what you don't remember."""
    rows = visible_sessions([session(id="hit"), session(id="miss", task="unrelated")],
                             query="importer fix")
    assert [r["id"] for r in rows] == ["hit"]


def test_the_search_also_matches_the_id_and_the_name():
    rows = visible_sessions([session(id="abc123", name="nightly")], query="abc")
    assert rows and visible_sessions(rows, query="NIGHTLY")


def test_every_word_has_to_match():
    assert visible_sessions([session()], query="importer banana") == []


# ---- the scroll window ----

def test_a_short_list_is_not_scrolled():
    assert scroll_window(0, 3, rows=6) == (0, 3)


def test_the_window_follows_the_selection_down():
    first, last = scroll_window(20, 40, rows=6)
    assert first <= 20 < last and last - first == 6


def test_the_window_stops_at_the_end_of_the_list():
    assert scroll_window(39, 40, rows=6) == (34, 40)


# ---- the picker's state ----

def test_arrows_clamp_at_both_ends():
    picker = SessionPicker([session(id="a"), session(id="b")], "/work/omni")
    picker.move(-5)
    assert picker.selected["id"] == "a"
    picker.move(99)
    assert picker.selected["id"] == "b"


def test_typing_puts_the_selection_back_on_the_first_match():
    """Leaving it where it was means Enter resumes whichever session happens
    to sit at that offset in a list you have just replaced."""
    picker = SessionPicker([session(id="a"), session(id="b", task="something else")],
                            "/work/omni")
    picker.move(1)
    picker.search("something")
    assert picker.index == 0 and picker.selected["id"] == "b"


def test_ctrl_a_widens_to_every_project_and_keeps_the_selection():
    rows = [session(id="a"), session(id="elsewhere", project_root="/work/other")]
    picker = SessionPicker(rows, "/work/omni")
    assert [r["id"] for r in picker.visible] == ["a"]
    picker.toggle_projects()
    assert len(picker.visible) == 2
    assert picker.selected["id"] == "a"


def test_ctrl_t_widens_to_every_branch_and_keeps_the_selection():
    rows = [session(id="a"), session(id="other", branch="feature/x")]
    picker = SessionPicker(rows, "/work/omni", "master")
    assert [r["id"] for r in picker.visible] == ["a"]
    picker.toggle_branches()
    assert len(picker.visible) == 2
    assert picker.selected["id"] == "a"


def test_the_branch_toggle_does_nothing_outside_a_checkout():
    """There is no branch to scope to, so the key must not silently flip a
    filter that isn't doing anything."""
    picker = SessionPicker([session()], "/work/omni", branch="")
    picker.toggle_branches()
    assert picker.all_branches and len(picker.visible) == 1


def test_a_project_with_no_sessions_opens_on_every_project():
    """An empty list plus a hint to press a key is a worse first screen than
    simply showing what there is."""
    picker = SessionPicker([session(project_root="/work/other")], "/work/omni")
    assert picker.all_projects and len(picker.visible) == 1


def test_a_branch_with_no_sessions_opens_on_every_branch():
    picker = SessionPicker([session(branch="feature/x")], "/work/omni", "master")
    assert picker.all_branches and len(picker.visible) == 1


def test_the_branch_is_widened_before_the_project():
    """The branch is the narrower of the two and far more often the reason
    the list came up empty, so widening it is the smaller step."""
    picker = SessionPicker([session(branch="feature/x")], "/work/omni", "master")
    assert picker.all_branches and not picker.all_projects


def test_selected_is_none_when_nothing_matches():
    picker = SessionPicker([session()], "/work/omni")
    picker.search("no such thing")
    assert picker.selected is None


# ---- how it looks ----

def test_the_heading_counts_the_filtered_list_not_the_whole_one():
    picker = SessionPicker([session(id="a"), session(id="b", task="other")], "/work/omni")
    picker.search("importer")
    assert text(picker.title_fragments()) == "  Resume session  (1 of 1)"


def test_the_heading_names_the_project_and_the_branch():
    picker = SessionPicker([session()], "/work/omni", "master")
    assert text(picker.scope_fragments()).strip() == "omni  ·  master"
    picker.toggle_projects()
    picker.toggle_branches()
    assert text(picker.scope_fragments()).strip() == "All projects  ·  all branches"


def test_the_heading_leaves_the_branch_out_when_there_is_none():
    picker = SessionPicker([session()], "/work/omni", branch="")
    assert text(picker.scope_fragments()).strip() == "omni"


def test_the_selected_row_is_marked_and_accented():
    picker = SessionPicker([session(id="a"), session(id="b", name="second")], "/work/omni")
    picker.move(1)
    fragments = picker.row_fragments(NOW)
    assert ("class:picker.marker", "❯ ") in fragments
    assert ("class:picker.name.selected", "second") in fragments
    assert ("class:picker.name", "fix the importer") in fragments


def test_a_scrolled_list_says_which_way_it_runs():
    picker = SessionPicker([session(id=str(i), name=f"s{i}") for i in range(20)],
                            "/work/omni", rows=6)
    picker.move(10)
    assert "↑ " in text(picker.row_fragments(NOW))
    assert "↓ " in text(picker.row_fragments(NOW))


def test_an_empty_result_says_so_instead_of_drawing_nothing():
    picker = SessionPicker([session()], "/work/omni")
    picker.search("no such thing")
    assert "Nothing matches" in text(picker.row_fragments(NOW))


def test_every_style_the_picker_uses_is_in_the_theme():
    """A class the style map doesn't define renders as unstyled default
    text, which is how a themed list quietly turns grey."""
    picker = SessionPicker([session(id=str(i)) for i in range(20)], "/work/omni", rows=6)
    picker.move(3)
    used = set()
    for build in (picker.title_fragments, picker.scope_fragments,
                   picker.hint_fragments, lambda: picker.row_fragments(NOW)):
        used |= {s for s in styles(build()) if s}
    known = {f"class:{name}" for names, _ in ui._build_prompt_style().class_names_and_attrs
              for name in names}
    assert used <= known


def test_the_accent_reaches_the_picker():
    """--theme-color has to recolour this list too, not leave one rust
    heading behind."""
    ui.set_accent("#00b4d8")
    try:
        style = dict(ui._build_prompt_style().class_names_and_attrs)
        assert "00b4d8" in str(style[frozenset({"picker.title"})])
        assert "00b4d8" in str(style[frozenset({"picker.name.selected"})])
    finally:
        ui.set_accent(ui.DEFAULT_ACCENT)


def test_the_hint_offers_ctrl_t_for_branches_and_never_ctrl_b():
    """Ctrl+B is the Windows image-paste key — Windows Terminal eats Ctrl+V
    — so the branch toggle must not claim it."""
    picker = SessionPicker([session()], "/work/omni", "master")
    hint = text(picker.hint_fragments())
    assert "Ctrl+T for all branches" in hint
    assert "Ctrl+A for all projects" in hint
    for taken in ("Ctrl+B", "Ctrl+V", "Ctrl+S", "Ctrl+C"):
        assert taken not in hint
    assert "Enter to resume" in hint and "Esc to cancel" in hint


def test_the_hint_drops_the_branch_toggle_outside_a_checkout():
    """Offering a key that does nothing is worse than not offering it."""
    hint = text(SessionPicker([session()], "/work/omni", branch="").hint_fragments())
    assert "Ctrl+T" not in hint and "Ctrl+A" in hint


def test_the_picker_binds_nothing_the_session_already_uses():
    """The picker is its own Application, so prompt_toolkit would happily let
    it rebind Ctrl+V or Ctrl+B — and then one chord would mean paste
    everywhere else and something else here."""
    picker = SessionPicker([session()], "/work/omni")
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            bound = {
                "-".join(str(key) for key in binding.keys)
                for binding in picker._build().key_bindings.bindings
            }
    assert bound and not (bound & RESERVED_KEYS)


def test_the_keys_the_picker_reserves_are_the_ones_the_session_really_binds():
    """RESERVED_KEYS is a hand-written list; this is what keeps it honest as
    the full-screen UI grows new chords."""
    import re
    from pathlib import Path
    source = Path(__file__).resolve().parent.parent / "omni" / "tui.py"
    bound = set(re.findall(r'\.add\("(c-[a-z]+)"', source.read_text(encoding="utf-8")))
    # c-c and c-d mean "stop"/"leave" in both places, so the picker reuses
    # them on purpose; everything else the session binds is off limits.
    assert bound - {"c-c", "c-d"} == set(RESERVED_KEYS)


def test_the_hint_names_the_scope_each_toggle_would_move_to():
    picker = SessionPicker([session()], "/work/omni", "master")
    picker.toggle_projects()
    picker.toggle_branches()
    hint = text(picker.hint_fragments())
    assert "Ctrl+A for only this project" in hint
    assert "Ctrl+T for only this branch" in hint


# ---- driving it ----

def drive(picker, keys: str):
    """Run the picker against a pipe, feeding it `keys`, and return what it
    selected."""
    async def go():
        with create_pipe_input() as pipe:
            with create_app_session(input=pipe, output=DummyOutput()):
                picker._app = picker._build()
                pipe.send_text(keys)
                try:
                    return await picker._app.run_async()
                finally:
                    picker._app = None
    return asyncio.run(go())


DOWN, UP, ENTER, ESC = "\x1b[B", "\x1b[A", "\r", "\x1b"
CTRL_A, CTRL_T = "\x01", "\x14"


def test_arrows_and_enter_pick_a_session():
    picker = SessionPicker([session(id=f"s{i}", name=f"row {i}") for i in range(5)], "/work/omni")
    assert drive(picker, DOWN + DOWN + UP + ENTER) == "s1"


def test_typing_narrows_the_list_before_enter_takes_it():
    rows = [session(id="a", name="importer"), session(id="b", name="exporter", task="")]
    assert drive(SessionPicker(rows, "/work/omni"), "export" + ENTER) == "b"


def test_escape_cancels_without_picking_anything():
    assert drive(SessionPicker([session(id="a")], "/work/omni"), ESC) is None


def test_ctrl_a_widens_to_every_project_while_the_list_is_open():
    rows = [session(id="here"), session(id="there", project_root="/work/other", name="zzz")]
    assert drive(SessionPicker(rows, "/work/omni"), CTRL_A + DOWN + ENTER) == "there"


def test_ctrl_t_widens_to_every_branch_while_the_list_is_open():
    rows = [session(id="here"), session(id="there", branch="feature/x", name="zzz")]
    picker = SessionPicker(rows, "/work/omni", "master")
    assert drive(picker, CTRL_T + DOWN + ENTER) == "there"


def test_enter_on_an_empty_result_does_nothing():
    """No selection means no id to return, so Enter must not exit with the
    previous highlight — or with a crash."""
    picker = SessionPicker([session(id="a")], "/work/omni")
    assert drive(picker, "no such thing" + ENTER + ESC) is None
