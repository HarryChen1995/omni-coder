"""The full-screen UI: the transcript's layout, scrolling and click handling.

These are the parts that make clicking possible, so they're tested directly
rather than through a terminal: a Block renders at a width, the Transcript
maps a click's row back to the block that drew it, and toggling one keeps
everything else where it was. Driving a real terminal (mouse escapes and all)
is left to the pty checks.
"""

import pytest
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from rich.console import Group
from rich.text import Text

from omni.tui import Block, Transcript


def click(row: int, kind=MouseEventType.MOUSE_UP) -> MouseEvent:
    return MouseEvent(position=Point(x=2, y=row), event_type=kind,
                       button=MouseButton.LEFT, modifiers=frozenset())


def lines_of(transcript: Transcript) -> list:
    return ["".join(fragment[1] for fragment in row[0]) for row in transcript._rows]


def plain(text: str) -> Block:
    return Block(collapsed=lambda: Text(text))


def foldable(label: str, opened: int = 3) -> Block:
    return Block(collapsed=lambda: Text(f"{label} collapsed"),
                  expanded=lambda: Group(*[Text(f"{label} open {i}") for i in range(opened)]))


@pytest.fixture
def transcript():
    t = Transcript()
    t.add(foldable("A"))
    t.add(plain("B plain"))
    t.add(foldable("C", opened=2))
    t.create_content(60, 12)
    return t


# ---------------- layout ----------------

def test_blocks_lay_out_one_after_another(transcript):
    assert lines_of(transcript) == ["A collapsed", "B plain", "C collapsed"]


def test_a_block_re_renders_when_the_width_changes():
    block = Block(collapsed=lambda: Text("x" * 100))
    narrow = block.lines(20)
    wide = block.lines(200)
    assert len(narrow) > len(wide) == 1


def test_rendering_is_cached_per_width_and_state():
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return Text("hello")

    block = Block(collapsed=build)
    block.lines(40)
    block.lines(40)
    assert calls["n"] == 1          # second call served from the cache
    block.lines(41)
    assert calls["n"] == 2          # a new width has to be rendered


# ---------------- clicking ----------------

def test_clicking_a_block_expands_it_in_place(transcript):
    transcript.mouse_handler(click(0))
    transcript.create_content(60, 12)
    assert lines_of(transcript) == ["A open 0", "A open 1", "A open 2", "B plain", "C collapsed"]


def test_clicking_an_open_block_collapses_it_again(transcript):
    transcript.mouse_handler(click(0))
    transcript.create_content(60, 12)
    transcript.mouse_handler(click(1))          # any row of the open block
    transcript.create_content(60, 12)
    assert lines_of(transcript) == ["A collapsed", "B plain", "C collapsed"]


def test_clicking_a_plain_block_does_nothing(transcript):
    before = lines_of(transcript)
    assert transcript.mouse_handler(click(1)) is NotImplemented
    transcript.create_content(60, 12)
    assert lines_of(transcript) == before


def test_clicking_past_the_end_is_ignored(transcript):
    assert transcript.mouse_handler(click(50)) is NotImplemented


def test_each_block_answers_for_its_own_rows(transcript):
    transcript.mouse_handler(click(2))          # the third block
    transcript.create_content(60, 12)
    assert lines_of(transcript) == ["A collapsed", "B plain", "C open 0", "C open 1"]


def test_only_the_clicked_block_changes(transcript):
    transcript.mouse_handler(click(0))
    transcript.mouse_handler(click(0))
    transcript.create_content(60, 12)
    assert transcript.blocks[1].is_open is False and transcript.blocks[2].is_open is False


# ---------------- scrolling ----------------

@pytest.fixture
def long_transcript():
    t = Transcript()
    for i in range(30):
        t.add(plain(f"line {i}"))
    t.create_content(60, 5)
    return t


def test_new_output_pins_the_view_to_the_bottom(long_transcript):
    assert long_transcript.follow is True
    assert long_transcript.scroll == long_transcript.max_scroll(5)


def test_the_wheel_scrolls_up_and_stops_following(long_transcript):
    long_transcript.mouse_handler(click(0, MouseEventType.SCROLL_UP))
    long_transcript.create_content(60, 5)
    assert long_transcript.scroll < long_transcript.max_scroll(5)
    assert long_transcript.follow is False


def test_scrolling_back_to_the_bottom_resumes_following(long_transcript):
    long_transcript.mouse_handler(click(0, MouseEventType.SCROLL_UP))
    for _ in range(5):
        long_transcript.mouse_handler(click(0, MouseEventType.SCROLL_DOWN))
    long_transcript.create_content(60, 5)
    assert long_transcript.follow is True


def test_scroll_never_leaves_the_content(long_transcript):
    for _ in range(50):
        long_transcript.mouse_handler(click(0, MouseEventType.SCROLL_UP))
    assert long_transcript.scroll == 0
    for _ in range(200):
        long_transcript.mouse_handler(click(0, MouseEventType.SCROLL_DOWN))
    assert long_transcript.scroll == long_transcript.max_scroll(5)


def test_a_click_maps_through_the_scroll_offset():
    t = Transcript()
    for i in range(10):
        t.add(plain(f"filler {i}"))
    target = t.add(foldable("target"))
    t.create_content(60, 4)             # scrolled to the bottom
    t.mouse_handler(click(3))           # the last visible row is the target
    t.create_content(60, 4)
    assert target.is_open is True


# ---------------- the application ----------------
#
# The frame has one input row and four states behind it (idle, busy, approve,
# ask), so these check that a keystroke lands wherever the current state says
# it should — that is the whole reason there is only ever one place to type.

import asyncio

from omni import ui
from omni.tui import TuiApp


@pytest.fixture
def app(mocker):
    mocker.patch("omni.ui.instruction")           # would emit into a transcript block
    application = TuiApp({"/exit": "leave"}, "my-session", "my-model")
    ui.use_tui(application)
    yield application
    ui.use_tui(None)


def binding(application, key, *, filter_mode=None):
    """The handler bound to `key` that is enabled in the app's current mode."""
    from prompt_toolkit.keys import Keys
    wanted = {"enter": Keys.ControlM, "c-c": Keys.ControlC, "c-d": Keys.ControlD,
              "y": "y", "n": "n", "pageup": Keys.PageUp,
              "up": Keys.Up, "down": Keys.Down}[key]
    for b in application._app.key_bindings.bindings:
        if b.keys == (wanted,) and b.filter():
            return b.handler
    raise AssertionError(f"no active binding for {key} in mode {application.mode}")


def test_enter_hands_the_instruction_over(app, mocker):
    app._buffer.text = "do the thing"
    binding(app, "enter")(mocker.Mock())
    pane, text = app._queue.get_nowait()
    assert (pane, text) == (app.main, "do the thing")
    assert app._buffer.text == ""


def test_enter_echoes_the_instruction_into_the_transcript(app, mocker):
    echo = mocker.patch("omni.ui.instruction")
    app._buffer.text = "visible please"
    binding(app, "enter")(mocker.Mock())
    echo.assert_called_once_with("visible please", images=0)


def test_a_blank_line_is_not_an_instruction(app, mocker):
    app._buffer.text = "   "
    binding(app, "enter")(mocker.Mock())
    assert app._queue.empty()


def test_ctrl_c_clears_the_line_when_idle(app, mocker):
    app._buffer.text = "half typed"
    binding(app, "c-c")(mocker.Mock())
    assert app._buffer.text == "" and app._queue.empty()


def test_ctrl_c_interrupts_the_turn_when_busy(app, mocker):
    cancel = mocker.Mock()
    app.pane.on_interrupt = cancel      # interrupts belong to a pane, not the window
    app.set_busy("Thinking…")
    binding(app, "c-c")(mocker.Mock())
    cancel.assert_called_once()


def test_ctrl_d_on_an_empty_line_ends_the_session(app, mocker):
    binding(app, "c-d")(mocker.Mock())
    assert app._queue.get_nowait() == (None, None)


def test_ctrl_d_with_text_keeps_the_session(app, mocker):
    app._buffer.text = "keep me"
    binding(app, "c-d")(mocker.Mock())
    assert app._queue.empty()


def test_busy_status_line_advances(app):
    app.set_busy("Running read_file…")
    first = app._status_line()
    assert "read_file" in first[1][1]
    assert app._status_line()[2][1].endswith("s)")   # elapsed counter


def test_set_label_relabels_without_leaving_busy(app):
    app.set_busy("Thinking…")
    app.set_label("Running write_file…")
    assert app.mode == "busy" and "write_file" in app._status_line()[1][1]


async def test_approval_is_answered_in_the_frame(app, mocker):
    pending = asyncio.ensure_future(app.ask_approval("Approve write_file?"))
    await asyncio.sleep(0)
    assert app.mode == "approve" and "write_file" in app._approve_line()[1][1]
    binding(app, "y")(mocker.Mock())
    assert await pending is True
    assert app.mode == "idle"                        # restored


async def test_approval_defaults_to_no_on_enter(app, mocker):
    pending = asyncio.ensure_future(app.ask_approval("Approve run_shell?"))
    await asyncio.sleep(0)
    binding(app, "enter")(mocker.Mock())
    assert await pending is False


async def test_approval_is_declined_by_ctrl_c(app, mocker):
    pending = asyncio.ensure_future(app.ask_approval("Approve run_shell?"))
    await asyncio.sleep(0)
    binding(app, "c-c")(mocker.Mock())
    assert await pending is False


async def test_a_question_reads_a_line_without_starting_a_turn(app, mocker):
    """ask_user's answer goes back to the tool, not into the instruction
    queue — same input row, different destination."""
    pending = asyncio.ensure_future(app.ask_text("type 1–2 to choose"))
    await asyncio.sleep(0)
    assert app.mode == "ask"
    app._buffer.text = "2"
    binding(app, "enter")(mocker.Mock())
    assert await pending == "2"
    assert app._queue.empty() and app.mode == "idle"


async def test_options_are_shown_as_a_picker_with_a_highlight(app, mocker):
    pending = asyncio.ensure_future(app.ask_text("choose", ["Keep SQLite", "Switch to Postgres"]))
    await asyncio.sleep(0)
    rows = "".join(f[1] for f in app._options_lines())
    assert "1. Keep SQLite" in rows and "2. Switch to Postgres" in rows
    assert rows.strip().startswith("▸")               # the first is highlighted
    binding(app, "enter")(mocker.Mock())
    assert await pending == "Keep SQLite"             # highlight submits as the answer


async def test_arrow_keys_move_the_highlight(app, mocker):
    pending = asyncio.ensure_future(app.ask_text("choose", ["a", "b", "c"]))
    await asyncio.sleep(0)
    binding(app, "down")(mocker.Mock())
    binding(app, "down")(mocker.Mock())
    binding(app, "enter")(mocker.Mock())
    assert await pending == "c"


async def test_the_highlight_wraps_around(app, mocker):
    pending = asyncio.ensure_future(app.ask_text("choose", ["a", "b"]))
    await asyncio.sleep(0)
    binding(app, "up")(mocker.Mock())                 # up from the first
    binding(app, "enter")(mocker.Mock())
    assert await pending == "b"


async def test_typing_beats_the_highlight(app, mocker):
    """The interesting answer is often none of the options, so a typed line
    always wins — and the picker says so while you type."""
    pending = asyncio.ensure_future(app.ask_text("choose", ["a", "b"]))
    await asyncio.sleep(0)
    app._buffer.text = "neither, use DuckDB"
    assert "using what you typed" in "".join(f[1] for f in app._options_lines())
    binding(app, "enter")(mocker.Mock())
    assert await pending == "neither, use DuckDB"


async def test_an_option_can_be_clicked(app, mocker):
    from prompt_toolkit.mouse_events import MouseEventType
    pending = asyncio.ensure_future(app.ask_text("choose", ["a", "b", "c"]))
    await asyncio.sleep(0)
    fragments = app._options_lines()
    handler = next(f[2] for f in fragments if len(f) == 3 and "3. c" in f[1])
    handler(mocker.Mock(event_type=MouseEventType.MOUSE_UP))
    binding(app, "enter")(mocker.Mock())
    assert await pending == "c"


async def test_the_picker_is_gone_once_answered(app, mocker):
    pending = asyncio.ensure_future(app.ask_text("choose", ["a"]))
    await asyncio.sleep(0)
    binding(app, "enter")(mocker.Mock())
    await pending
    assert app.pane.options == [] and app._choosing() is False


async def test_a_question_restores_the_busy_state_afterwards(app, mocker):
    app.set_busy("Thinking…")
    pending = asyncio.ensure_future(app.ask_text("answer"))
    await asyncio.sleep(0)
    app._buffer.text = "ok"
    binding(app, "enter")(mocker.Mock())
    await pending
    assert app.mode == "busy"


async def test_dismissing_a_question_answers_none(app, mocker):
    pending = asyncio.ensure_future(app.ask_text("answer"))
    await asyncio.sleep(0)
    binding(app, "c-c")(mocker.Mock())
    assert await pending is None


def test_the_frame_carries_the_session_chip_and_model(app):
    assert " my-session " in "".join(t for _, t in app._rule_with_chip())
    assert "my-model" in "".join(t for _, t in app._hint())


def test_the_hint_follows_the_mode(app):
    idle = "".join(t for _, t in app._hint())
    app.set_busy("Thinking…")
    busy = "".join(t for _, t in app._hint())
    assert "click ▸ to expand" in idle and "ctrl+c interrupts" in busy


def test_emit_adds_a_transcript_block(app):
    app.emit(lambda: Text("hello"))
    assert len(app.pane.transcript.blocks) == 1


def test_dump_writes_every_block(app, capsys):
    app.emit(lambda: Text("first"))
    app.emit(lambda: Text("second"), lambda: Text("second expanded"))
    app.dump()
    out = capsys.readouterr().out
    assert "first" in out and "second" in out


def test_dump_writes_open_blocks_in_their_open_form(app, capsys):
    block = app.emit(lambda: Text("shut"), lambda: Text("opened"))
    block.is_open = True
    app.dump()
    assert "opened" in capsys.readouterr().out


# ---------------- panes: several agents at once ----------------
#
# Main is pane 0; every subagent gets its own. What matters here is that
# output, questions and interrupts land on the pane that owns them even while
# three agents are running — which is what the ContextVar is for — and that
# the tree reflects each one's state.

from omni.tui import Pane, current_pane


def test_a_session_starts_with_only_main(app):
    assert [p.kind for p in app.panes] == ["main"]
    assert app.main is app.pane and app.pane.depth == 0


def test_adding_a_subagent_gives_it_its_own_transcript(app):
    sub = app.add_pane("audit", depth=1)
    assert sub.kind == "subagent" and sub.transcript is not app.main.transcript
    assert app.panes == [app.main, sub]
    assert app.pane is app.main          # spawning doesn't steal focus


def test_output_goes_to_the_pane_that_produced_it(app):
    sub = app.add_pane("audit", depth=1)
    token = current_pane.set(sub)
    try:
        app.emit(lambda: Text("from the subagent"))
    finally:
        current_pane.reset(token)
    assert len(sub.transcript.blocks) == 1
    assert app.main.transcript.blocks == []
    assert sub.unseen is True            # flagged, since you were elsewhere


def test_switching_clears_the_unseen_flag(app):
    sub = app.add_pane("audit", depth=1)
    sub.unseen = True
    app.focus(1)
    assert app.pane is sub and sub.unseen is False


def test_the_transcript_window_follows_focus(app):
    sub = app.add_pane("audit", depth=1)
    token = current_pane.set(sub)
    try:
        app.emit(lambda: Text("subagent line"))
    finally:
        current_pane.reset(token)
    app.focus(1)
    app.pane.transcript.create_content(60, 10)
    rows = lines_of(app.pane.transcript)
    assert any("subagent line" in row for row in rows)


async def test_background_approval_does_not_take_over_the_frame(app, mocker):
    sub = app.add_pane("writer", depth=1)
    pending = asyncio.ensure_future(app.ask_approval("Approve write_file?", pane=sub))
    await asyncio.sleep(0)
    assert app.mode == "idle"            # main's frame is untouched
    assert sub.needs_you is True
    assert "!" in "".join(f[1] for f in app._agent_tree())
    app.focus(1)
    assert app.mode == "approve"         # ...and there it is, once you switch
    binding(app, "y")(mocker.Mock())
    assert await pending is True


async def test_typing_goes_to_the_focused_pane(app, mocker):
    sub = app.add_pane("audit", depth=1)
    app.focus(1)
    app._buffer.text = "narrow it to save_memory"
    binding(app, "enter")(mocker.Mock())
    pane, text = app._queue.get_nowait()
    assert pane is sub and text == "narrow it to save_memory"


def test_status_and_tokens_are_per_pane(app, mocker):
    sub = app.add_pane("audit", agent=mocker.Mock(tokens={"prompt": 12100, "completion": 3400}),
                        depth=1)
    app.set_busy("Thinking…", pane=sub)
    assert app.main.busy is False and sub.busy is True
    app.focus(1)
    line = "".join(f[1] for f in app._status_line())
    assert "↑ 12.1k" in line and "↓ 3.4k" in line


# ---------------- the agent tree ----------------

async def test_main_is_labelled_main_and_subagents_are_numbered(app):
    app.add_pane("files", depth=1)
    app.add_pane("readme", depth=1)
    rows = "".join(f[1] for f in app._agent_tree())
    assert "main" in rows and "0 " not in rows      # main isn't a numbered subagent
    assert "1 files" in rows and "2 readme" in rows


async def test_the_tree_marks_each_state(app, mocker):
    working = app.add_pane("working", depth=1)
    finished = app.add_pane("finished", depth=1)
    waiting = app.add_pane("waiting", depth=1)
    app.set_busy("Thinking…", pane=working)
    finished.done = True
    waiting.approval = ("Approve?", asyncio.get_running_loop().create_future())

    rows = "".join(f[1] for f in app._agent_tree())
    assert "○" in rows          # working
    assert "●" in rows          # finished and reported back
    assert "!" in rows          # waiting on you


def test_a_tree_row_can_be_clicked(app, mocker):
    from prompt_toolkit.mouse_events import MouseEventType
    sub = app.add_pane("audit", depth=1)
    handler = next(f[2] for f in app._agent_tree() if len(f) == 3 and "1 audit" in f[1])
    handler(mocker.Mock(event_type=MouseEventType.MOUSE_UP))
    assert app.pane is sub


def test_arrows_walk_the_tree_when_the_prompt_is_empty(app, mocker):
    """The tree is visible under the input, so that's where the arrows go —
    until you type something, when they belong to the line again."""
    sub = app.add_pane("audit", depth=1)
    assert app._navigating() is True
    binding(app, "down")(mocker.Mock())
    assert app._picking is True and app._pick_index == 1
    binding(app, "enter")(mocker.Mock())
    assert app.pane is sub and app._picking is False


def test_the_tree_can_be_walked_while_an_agent_is_working(app, mocker):
    """When main is busy is exactly when you want to look at a subagent — and
    getting this wrong sent ctrl+c to the wrong agent."""
    sub = app.add_pane("audit", depth=1)
    app.set_busy("Thinking…", pane=app.main)
    assert app._navigating() is True
    binding(app, "down")(mocker.Mock())
    binding(app, "enter")(mocker.Mock())
    assert app.pane is sub


async def test_a_question_keeps_the_arrows(app, mocker):
    """A choice picker owns ↑↓ and Enter while it is up."""
    app.add_pane("audit", depth=1)
    pending = asyncio.ensure_future(app.ask_text("choose", ["a", "b"]))
    await asyncio.sleep(0)
    assert app._navigating() is False
    binding(app, "enter")(mocker.Mock())
    assert await pending == "a"


def test_typing_takes_the_arrows_back(app):
    app.add_pane("audit", depth=1)
    app._buffer.text = "half typed"
    assert app._navigating() is False


def test_a_lone_main_agent_shows_no_tree(app):
    assert len(app.panes) == 1        # the tree's container is filtered off


# ---------------- folding finished subagents ----------------

def test_finished_subagents_fold_into_main(app):
    """Their panes disappear once they've all reported back, but the work
    stays reachable as one clickable block in main."""
    sub = app.add_pane("audit", depth=1)
    token = current_pane.set(sub)
    try:
        app.emit(lambda: Text("what the subagent did"))
    finally:
        current_pane.reset(token)
    app.set_idle(pane=sub, done=True)
    app._fold_finished()

    assert app.panes == [app.main]
    assert len(app.main.transcript.blocks) == 1
    block = app.main.transcript.blocks[0]
    assert block.clickable
    assert "what the subagent did" in block.expanded()
    assert "audit" in block.collapsed()


def test_folding_waits_for_the_last_subagent(app):
    first = app.add_pane("one", depth=1)
    second = app.add_pane("two", depth=1)
    app.set_idle(pane=first, done=True)
    assert app._all_subagents_finished() is False    # `two` is still going
    app.set_idle(pane=second, done=True)
    assert app._all_subagents_finished() is True
    app._fold_finished()
    assert app.panes == [app.main]                   # both folded, together


def test_a_finished_subagent_shows_green_before_folding(app):
    sub = app.add_pane("audit", depth=1)
    app.set_idle(pane=sub, done=True)
    assert sub.done is True
    assert "●" in "".join(f[1] for f in app._agent_tree())


def test_closing_a_pane_by_hand(app):
    sub = app.add_pane("audit", depth=1)
    app.focus(1)
    assert app.close_pane(sub) is True
    assert app.panes == [app.main] and app.pane is app.main


def test_main_cannot_be_closed(app):
    assert app.close_pane(app.main) is False
