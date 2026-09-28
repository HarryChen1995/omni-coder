"""The choice picker an `ask_user` question puts up when there is no
full-screen frame to draw the choices in.

Same shape as the --resume list's tests: the state and the rows are checked
directly, and the keys are driven through a pipe with a dummy output, so
nothing here needs a real terminal.
"""

import asyncio
import sys

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from omni import ui
from omni.option_picker import OptionPicker, available
from omni.session_picker import RESERVED_KEYS

OPTIONS = ["Keep SQLite", "Switch to Postgres", "Neither"]


def styles(fragments):
    return [style for style, _ in fragments]


def text(fragments):
    return "".join(body for _, body in fragments)


def rows(picker) -> str:
    return text(picker.row_fragments())


# ---- what is highlighted ----

def test_the_first_choice_is_highlighted_to_begin_with():
    """The question is answerable with one keystroke; that only reads as true
    if something is already marked when the list appears."""
    assert rows(OptionPicker(OPTIONS)).startswith("  ❯ 1. Keep SQLite")


def test_arrows_move_the_marker():
    picker = OptionPicker(OPTIONS)
    picker.move(1)
    assert "❯ 2. Switch to Postgres" in rows(picker)
    assert "❯ 1." not in rows(picker)


def test_exactly_one_row_is_ever_marked():
    picker = OptionPicker(OPTIONS)
    picker.move(1)
    assert rows(picker).count("❯") == 1


@pytest.mark.parametrize("delta, expected", [(-1, "Neither"), (3, "Keep SQLite")])
def test_moving_past_either_end_wraps(delta, expected):
    """Three choices all visible at once should not take four keystrokes to
    get from the last back to the first."""
    picker = OptionPicker(OPTIONS)
    picker.move(delta)
    assert picker.selected == expected


def test_moving_an_empty_list_is_harmless():
    picker = OptionPicker([])
    picker.move(1)
    assert picker.selected is None and picker.row_fragments() == []


# ---- the option the model would pick itself ----

def test_the_cursor_starts_on_the_recommendation():
    """Said twice, and this is the half that costs nothing to act on: Enter
    takes the recommendation without moving at all."""
    picker = OptionPicker(OPTIONS, recommended=1)
    assert picker.index == 1 and picker.answer() == "Switch to Postgres"


def test_the_recommended_row_is_labelled():
    """The other half: a cursor that merely starts somewhere doesn't say that
    somewhere was chosen for a reason."""
    assert "2. Switch to Postgres  (recommended)" in rows(OptionPicker(OPTIONS, recommended=1))


def test_the_label_stays_behind_when_the_cursor_moves_off_it():
    """The recommendation belongs to the option, not to the cursor — and you
    have to be able to find your way back to it."""
    picker = OptionPicker(OPTIONS, recommended=1)
    picker.move(1)
    shown = rows(picker)
    assert "❯ 3. Neither" in shown
    assert "2. Switch to Postgres  (recommended)" in shown


def test_only_one_row_is_ever_recommended():
    assert rows(OptionPicker(OPTIONS, recommended=0)).count("(recommended)") == 1


def test_no_recommendation_means_no_label_and_the_first_row():
    picker = OptionPicker(OPTIONS)
    assert picker.recommended is None and picker.index == 0
    assert "(recommended)" not in rows(picker)


@pytest.mark.parametrize("junk", [-1, 3, 99])
def test_a_recommendation_naming_no_row_is_dropped_rather_than_crashing(junk):
    """ui normalises before this, but the picker is constructible on its own
    and an index past the end would blow up the first render."""
    picker = OptionPicker(OPTIONS, recommended=junk)
    assert picker.recommended is None and picker.index == 0
    assert "(recommended)" not in rows(picker)


def test_typing_hides_the_cursor_but_not_the_recommendation():
    """What the model suggested is still true while you write past it."""
    picker = OptionPicker(OPTIONS, recommended=1)
    picker.write("use DuckDB")
    shown = rows(picker)
    assert "❯" not in shown and "(recommended)" in shown


def test_enter_on_an_untouched_recommended_picker_takes_it():
    assert drive(OptionPicker(OPTIONS, recommended=2), ENTER) == "Neither"


def test_the_recommendation_can_still_be_arrowed_away_from():
    """A recommendation is a suggestion, not a default you are stuck with."""
    assert drive(OptionPicker(OPTIONS, recommended=1), UP + ENTER) == "Keep SQLite"


# ---- typing instead of choosing ----

def test_typing_takes_the_marker_off_the_list_and_says_why():
    """A marked row beside half-written text would claim Enter is about to
    send the row."""
    picker = OptionPicker(OPTIONS)
    picker.write("use DuckDB instead")
    assert "❯" not in rows(picker)
    assert "(using what you typed)" in rows(picker)


def test_whitespace_alone_is_not_typing():
    picker = OptionPicker(OPTIONS)
    picker.write("   ")
    assert not picker.typing
    assert "❯ 1. Keep SQLite" in rows(picker)


def test_clearing_what_was_typed_gives_the_marker_back():
    picker = OptionPicker(OPTIONS)
    picker.write("something")
    picker.write("")
    assert "❯ 1. Keep SQLite" in rows(picker)


# ---- the answer a state resolves to ----

def test_the_answer_is_the_highlighted_choice():
    picker = OptionPicker(OPTIONS)
    picker.move(2)
    assert picker.answer() == "Neither"


def test_what_was_typed_beats_what_is_highlighted():
    """The interesting answer is often none of the offered ones."""
    picker = OptionPicker(OPTIONS)
    picker.move(1)
    picker.write("use DuckDB instead")
    assert picker.answer() == "use DuckDB instead"


def test_the_typed_answer_is_stripped():
    picker = OptionPicker(OPTIONS)
    picker.write("  DuckDB  ")
    assert picker.answer() == "DuckDB"


def test_a_typed_number_is_left_as_text_for_the_caller_to_map():
    """ui.ask_user turns it into the option it names, so typing 2 and
    arrowing to the second row end up meaning the same thing."""
    picker = OptionPicker(OPTIONS)
    picker.write("2")
    assert picker.answer() == "2"


def test_an_empty_picker_answers_nothing():
    assert OptionPicker([]).answer() == ""


# ---- scrolling ----

def test_a_long_list_scrolls_and_keeps_the_selection_on_screen():
    picker = OptionPicker([f"option {i}" for i in range(20)], rows=4)
    picker.move(10)
    shown = rows(picker)
    assert "❯ 11. option 10" in shown
    assert len(shown.rstrip("\n").split("\n")) == 4


def test_a_scrolled_list_marks_the_edges_it_continues_past():
    picker = OptionPicker([f"option {i}" for i in range(20)], rows=4)
    picker.move(10)
    shown = rows(picker)
    assert "↑" in shown and "↓" in shown


def test_a_short_list_needs_no_scroll_markers():
    shown = rows(OptionPicker(OPTIONS))
    assert "↑" not in shown and "↓" not in shown


# ---- how it looks ----

def test_every_style_the_picker_uses_is_one_the_theme_defines():
    """A class the style map doesn't define renders as unstyled default text,
    which is how a themed list quietly turns grey."""
    picker = OptionPicker([f"option {i}" for i in range(20)], rows=4)
    picker.move(10)
    used = {s for s in styles(picker.row_fragments()) + styles(picker.hint_fragments()) if s}
    picker.write("typed")
    used |= {s for s in styles(picker.row_fragments()) if s}
    known = {f"class:{name}" for names, _ in ui._build_prompt_style().class_names_and_attrs
              for name in names}
    assert used <= known


def test_the_accent_reaches_the_marker():
    """--theme-color has to recolour this list too."""
    ui.set_accent("#00b4d8")
    try:
        style = dict(ui._build_prompt_style().class_names_and_attrs)
        assert "00b4d8" in str(style[frozenset({"picker.marker"})])
        assert "00b4d8" in str(style[frozenset({"picker.name.selected"})])
    finally:
        ui.set_accent(ui.DEFAULT_ACCENT)


def test_the_hint_names_the_keys_that_actually_do_something():
    hint = text(OptionPicker(OPTIONS).hint_fragments())
    for promised in ("↑↓ to move", "Enter to choose", "type your own", "Ctrl+C to dismiss"):
        assert promised in hint


def test_the_hint_claims_no_chord_the_session_already_uses():
    hint = text(OptionPicker(OPTIONS).hint_fragments())
    for taken in ("Ctrl+V", "Ctrl+B", "Ctrl+S", "Ctrl+A", "Ctrl+T"):
        assert taken not in hint


# ---- whether it can be drawn at all ----

class _Stream:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_a_terminal_on_both_ends_means_the_picker_can_run(mocker):
    mocker.patch.object(sys, "stdin", _Stream(True))
    mocker.patch.object(sys, "stdout", _Stream(True))
    assert available()


@pytest.mark.parametrize("stdin_tty, stdout_tty", [(False, True), (True, False), (False, False)])
def test_a_piped_run_has_no_arrow_keys_to_press(mocker, stdin_tty, stdout_tty):
    """`omni "…" < script.txt` must fall back to the numbered list rather than
    wait for a keystroke that is never coming."""
    mocker.patch.object(sys, "stdin", _Stream(stdin_tty))
    mocker.patch.object(sys, "stdout", _Stream(stdout_tty))
    assert not available()


def test_a_stream_that_cannot_answer_counts_as_no_terminal(mocker):
    """pytest's own capture replaces stdout with objects that raise on a
    closed buffer; a picker is not worth an exception."""
    mocker.patch.object(sys, "stdin", object())
    mocker.patch.object(sys, "stdout", _Stream(True))
    assert not available()


def test_availability_is_decided_the_same_way_on_every_platform():
    """macOS, Linux and Windows differ in what a terminal *is* underneath —
    a tty, or a console screen buffer — but nothing here may branch on which
    one it is running on. Whether prompt_toolkit can drive the thing is
    settled by trying (prepare), not by guessing from the platform."""
    from pathlib import Path
    source = Path(__file__).resolve().parent.parent / "omni" / "option_picker.py"
    body = source.read_text(encoding="utf-8")
    for guess in ("sys.platform", "os.name", "platform.system", "win32", "darwin"):
        assert guess not in body


# ---- whether the terminal will actually take it ----

def test_preparing_succeeds_where_prompt_toolkit_can_attach():
    picker = OptionPicker(OPTIONS)
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            assert picker.prepare()
    assert picker._app is not None


def test_preparing_fails_rather_than_raising_on_a_terminal_it_cannot_drive(mocker):
    """isatty() says yes on a Cygwin or mintty console prompt_toolkit then
    refuses to attach to, and on a POSIX terminal it cannot put into raw
    mode. Building the app is the step that finds out, so it is asked before
    the question is printed rather than after."""
    mocker.patch.object(OptionPicker, "_build", side_effect=RuntimeError("no console"))
    picker = OptionPicker(OPTIONS)
    assert not picker.prepare()
    assert picker._app is None


def test_running_an_unpreparable_picker_raises_for_the_caller_to_fall_back_on(mocker):
    """ui.ask_user catches this and asks the plain way; it must not come back
    as a silent None, which would read as the person dismissing the
    question."""
    mocker.patch.object(OptionPicker, "_build", side_effect=RuntimeError("no console"))
    with pytest.raises(RuntimeError):
        asyncio.run(OptionPicker(OPTIONS).run())


def test_preparing_then_running_builds_the_app_only_once(mocker):
    """The prepared app is the one the availability decision was made
    against; building a second at run time would be deciding twice."""
    async def go():
        with create_pipe_input() as pipe:
            with create_app_session(input=pipe, output=DummyOutput()):
                picker = OptionPicker(OPTIONS)
                spy = mocker.spy(OptionPicker, "_build")
                picker.prepare()
                pipe.send_text(ENTER)
                answer = await picker.run()
                return spy.call_count, answer
    assert asyncio.run(go()) == (1, "Keep SQLite")


def test_running_without_preparing_first_still_works():
    """`prepare` is an optimisation for the caller, not a required step."""
    async def go():
        with create_pipe_input() as pipe:
            with create_app_session(input=pipe, output=DummyOutput()):
                pipe.send_text(ENTER)
                return await OptionPicker(OPTIONS).run()
    assert asyncio.run(go()) == "Keep SQLite"


# ---- driving it ----

def drive(picker, keys: str):
    """Run the picker against a pipe, feeding it `keys`, and return its
    answer."""
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
CTRL_C = "\x03"


def test_arrows_and_enter_choose_an_option():
    """The whole point: move the cursor, press Enter, get that option."""
    assert drive(OptionPicker(OPTIONS), DOWN + DOWN + UP + ENTER) == "Switch to Postgres"


def test_enter_on_its_own_takes_the_first_option():
    assert drive(OptionPicker(OPTIONS), ENTER) == "Keep SQLite"


def test_arrowing_past_the_end_wraps_while_it_is_running():
    assert drive(OptionPicker(OPTIONS), UP + ENTER) == "Neither"


def test_typing_answers_past_the_list_entirely():
    assert drive(OptionPicker(OPTIONS), "use DuckDB instead" + ENTER) == "use DuckDB instead"


def test_a_number_typed_at_the_picker_comes_back_as_typed():
    assert drive(OptionPicker(OPTIONS), "2" + ENTER) == "2"


def test_what_was_typed_wins_even_after_arrowing():
    assert drive(OptionPicker(OPTIONS), DOWN + "neither, actually" + ENTER) == "neither, actually"


@pytest.mark.parametrize("key", [CTRL_C, ESC])
def test_dismissing_the_picker_answers_nothing(key):
    assert drive(OptionPicker(OPTIONS), key) is None


def test_enter_with_nothing_to_answer_does_not_exit_empty():
    """No options and no text means no answer; Enter must not resolve the
    question with an empty string."""
    assert drive(OptionPicker([]), ENTER + CTRL_C) is None


def test_the_picker_binds_nothing_the_session_already_uses():
    """It is its own Application, so prompt_toolkit would happily let it
    rebind Ctrl+V or Ctrl+B — and then one chord would mean paste everywhere
    else and something else here."""
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            bound = {
                "-".join(str(key) for key in binding.keys)
                for binding in OptionPicker(OPTIONS)._build().key_bindings.bindings
            }
    assert bound and not (bound & RESERVED_KEYS)


def test_the_picker_erases_itself_on_the_way_out():
    """The answer is echoed into the transcript by the caller; leaving the
    list behind as well would print the choices twice."""
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            app = OptionPicker(OPTIONS)._build()
    assert app.erase_when_done and not app.full_screen
