"""Full-screen terminal UI, so the transcript can be clicked.

A terminal owns everything already printed to it: text that has scrolled
cannot be rewritten, and mouse clicks are only ever delivered to the region an
application actually draws. Expanding a tool call *in place* therefore means
the transcript has to live inside the application rather than in the
terminal's scrollback — which is what this module is.

The pieces:

- `Block` — one entry in the transcript. Rich renderables, built lazily so a
  resize re-renders them, plus an optional expanded form. A block with an
  expanded form is clickable.
- `Transcript` — a UIControl that lays blocks out at the current width, keeps
  its own scroll position, and maps a click's row back to the block that drew
  it. Rich renders to ANSI; prompt_toolkit parses that back into fragments, so
  the two libraries meet at one well-defined seam.
- `TuiApp` — the whole screen: transcript above, and below it the same frame
  as before (rule with the session chip, the ❯ line, closing rule, hint),
  which doubles as the status line while a turn runs and as the y/n prompt
  when a tool call needs approval.

One-shot runs (`omni "task"`) don't use any of this; ui.py keeps printing
straight to the terminal there.
"""

import asyncio
import io
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    BufferControl, ConditionalContainer, FormattedTextControl, HSplit, Layout, VSplit, Window,
)
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.mouse_events import MouseEventType
from rich.console import Console

# ui.py, bound on first use. Imported lazily rather than at module scope so
# the two can reference each other without an import cycle.
_ui = None


def _bind_ui(module=None):
    global _ui
    if module is None:
        from . import ui as module
    _ui = module
    return _ui


def _ansi(renderable, width: int) -> str:
    """Render a Rich renderable to an ANSI string at `width`.

    force_terminal: the target is a real terminal, we're just not writing to
    it directly. truecolor keeps the accent exact rather than quantising it."""
    buf = io.StringIO()
    Console(file=buf, width=max(width, 20), force_terminal=True,
            color_system="truecolor", legacy_windows=False).print(renderable)
    return buf.getvalue()


@dataclass
class Block:
    """One transcript entry.

    `collapsed` and `expanded` are callables, not renderables: a resize has to
    re-render at the new width, and a tool call's expanded form is expensive
    enough (whole file contents) to be worth building only when opened."""

    collapsed: Callable[[], object]
    expanded: Optional[Callable[[], object]] = None
    is_open: bool = False
    # (width, is_open) -> the fragment lines that were produced for it
    _cache: tuple = field(default=None, repr=False)

    @property
    def clickable(self) -> bool:
        return self.expanded is not None

    def lines(self, width: int) -> list:
        if self._cache and self._cache[0] == (width, self.is_open):
            return self._cache[1]
        build = self.expanded if (self.is_open and self.expanded) else self.collapsed
        produced = build()
        text = produced if isinstance(produced, str) else _ansi(produced, width)
        lines = [line for line in split_lines(to_formatted_text(ANSI(text.rstrip("\n"))))]
        self._cache = ((width, self.is_open), lines)
        return lines

    def toggle(self) -> bool:
        if not self.clickable:
            return False
        self.is_open = not self.is_open
        return True


class Transcript(UIControl):
    """The scrollable, clickable transcript.

    Scrolling is handled here rather than by the Window: this control reports
    exactly the visible slice as its content, so prompt_toolkit never second-
    guesses the offset and a click's row maps to a block by simple arithmetic.
    """

    def __init__(self):
        self.blocks: list = []
        self.scroll = 0
        self.follow = True      # stay pinned to the newest output until scrolled up
        self._width = None
        self._rows: list = []   # one entry per laid-out row: (fragments, block index)
        self._signature = None

    # ---- content ----

    def add(self, block: Block) -> Block:
        self.blocks.append(block)
        self.follow = True      # new output pulls the view back to the bottom
        self._signature = None
        return block

    def clear(self):
        self.blocks.clear()
        self.scroll = 0
        self._signature = None

    def _layout(self, width: int):
        """(Re)build the row list. Cheap to call: it only redoes the work when
        the width or some block's open/closed state has changed."""
        signature = (width, tuple(b.is_open for b in self.blocks), len(self.blocks))
        if signature == self._signature:
            return
        rows = []
        for index, block in enumerate(self.blocks):
            for line in block.lines(width):
                rows.append((line, index))
        self._rows = rows
        self._signature = signature
        self._width = width

    def total_rows(self) -> int:
        return len(self._rows)

    def max_scroll(self, height: int) -> int:
        return max(len(self._rows) - height, 0)

    def create_content(self, width: int, height: int) -> UIContent:
        self._last_height = height   # the mouse handler needs it for scroll limits
        self._layout(width)
        if self.follow:
            self.scroll = self.max_scroll(height)
        else:
            self.scroll = min(self.scroll, self.max_scroll(height))
        visible = self._rows[self.scroll:self.scroll + height]

        def get_line(i: int):
            return visible[i][0] if i < len(visible) else []

        return UIContent(get_line=get_line, line_count=max(len(visible), 1),
                          cursor_position=Point(0, 0), show_cursor=False)

    def is_focusable(self) -> bool:
        return False   # focus belongs to the input line; mouse still arrives here

    # ---- scrolling ----

    def scroll_by(self, rows: int, height: int):
        self.scroll = max(0, min(self.scroll + rows, self.max_scroll(height)))
        self.follow = self.scroll >= self.max_scroll(height)

    def scroll_to_bottom(self):
        self.follow = True

    # ---- clicking ----

    def block_at_row(self, row: int) -> Optional[int]:
        """Which block drew the given *visible* row."""
        absolute = self.scroll + row
        if 0 <= absolute < len(self._rows):
            return self._rows[absolute][1]
        return None

    def toggle_at_row(self, row: int) -> bool:
        index = self.block_at_row(row)
        if index is None:
            return False
        block = self.blocks[index]
        if not block.clickable:
            return False
        # Keep the clicked block where it is on screen instead of letting the
        # rows below it push the view around as it grows.
        anchor = sum(len(b.lines(self._width)) for b in self.blocks[:index])
        block.toggle()
        self._signature = None
        self._layout(self._width)
        self.follow = False
        self.scroll = min(anchor, max(self.total_rows() - 1, 0))
        return True

    def mouse_handler(self, mouse_event):
        row = mouse_event.position.y
        height = self._last_height or 1
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.scroll_by(-3, height)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll_by(3, height)
            return None
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            if self.toggle_at_row(row):
                return None
        return NotImplemented

    _last_height = 1

    def preferred_height(self, width, max_available_height, wrap_lines, get_line_prefix):
        self._last_height = max_available_height
        return max_available_height


# --------------------------------------------------------------------------
# The application
# --------------------------------------------------------------------------

_DOTS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class TuiApp:
    """The whole screen for an interactive session.

    Transcript on top (scroll with the wheel, click a ▸ line to open it), the
    familiar frame underneath. The frame has three states, and only ever one
    at a time, so there is exactly one place to type:

      idle     ❯ and your text
      busy     what the agent is doing, with an elapsed counter
      approve  a y/n question about one tool call

    The REPL doesn't read input directly any more: this app runs for the whole
    session and hands instructions over a queue (`next_instruction`), which is
    what lets the transcript stay live and clickable while a turn runs.
    """

    def __init__(self, commands: dict, session_label: str = "", model: str = ""):
        if _ui is None:
            _bind_ui()
        self.transcript = Transcript()
        self.session_label = session_label
        self.model = model
        self.commands = commands
        self.on_interrupt = None          # set by the REPL while a turn runs

        self._mode = "idle"               # idle | busy | approve | ask
        self._ask = None                  # (hint, Future) while answering a question
        self._options: list = []          # choices offered with the question
        self._choice = 0                  # which one is highlighted
        self._status = ""
        self._phase_start = 0.0
        self._approval = None             # (question, Future)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._ticker = None
        self._exit_requested = False

        self._buffer = Buffer(
            completer=_ui.SlashCommandCompleter(commands),
            complete_while_typing=True,
            history=InMemoryHistory(),
            multiline=False,
        )
        self._app = Application(
            layout=self._build_layout(),
            key_bindings=self._build_keys(),
            style=_ui._PROMPT_STYLE,
            full_screen=True,
            mouse_support=True,
            erase_when_done=True,
        )

    # ---- state ----

    @property
    def mode(self) -> str:
        return self._mode

    def _idle(self) -> bool:
        return self._mode == "idle"

    def _busy(self) -> bool:
        return self._mode == "busy"

    def _approving(self) -> bool:
        return self._mode == "approve"

    def _asking(self) -> bool:
        return self._mode == "ask"

    def _choosing(self) -> bool:
        return self._asking() and bool(self._options)

    def _options_lines(self):
        """The choices, with the highlighted one marked. Each row carries its
        own mouse handler, so an option can be clicked as well as arrowed
        to — the transcript is clickable, and so is this."""
        typing = bool(self._buffer.text.strip())
        fragments = []
        for i, option in enumerate(self._options):
            selected = i == self._choice and not typing
            marker = "▸ " if selected else "  "
            style = f"class:choice.selected" if selected else "class:choice"

            def handler(event, index=i):
                if event.event_type == MouseEventType.MOUSE_UP:
                    self._choice = index
                    self._buffer.reset()
                    self.invalidate()

            fragments.append((style, f" {marker}{i + 1}. {option}", handler))
            fragments.append(("", "\n"))
        if typing:
            fragments.append(("class:frame.hint", "   (using what you typed)"))
        return fragments

    def _accepts_typing(self) -> bool:
        """Both idle and ask mode read a line from the input row — the
        difference is only where the line goes."""
        return self._idle() or self._asking()

    def invalidate(self):
        try:
            self._app.invalidate()
        except Exception:
            pass   # not running yet; the next render picks the state up anyway

    # ---- transcript ----

    def emit(self, collapsed, expanded=None) -> Block:
        """Add one entry. Pass `expanded` to make it clickable."""
        block = self.transcript.add(Block(collapsed=collapsed, expanded=expanded))
        self.invalidate()
        return block

    def transcript_width(self) -> int:
        """Width the transcript renders at — what ui.py renders its blocks to."""
        return self._width()

    def dump(self):
        """Write the transcript to the real terminal, in whatever open/closed
        state it was left in.

        A full-screen app hands the terminal back with its previous contents
        restored, which would otherwise mean the whole session vanishing from
        scrollback on exit. Written as raw ANSI rather than through Rich: the
        blocks are already rendered text, and re-printing them as markup
        would mangle the escapes they contain. Must be called while ui still
        has this app registered, since that is what the block closures render
        against."""
        width = self.transcript_width()
        for block in self.transcript.blocks:
            build = block.expanded if (block.is_open and block.expanded) else block.collapsed
            try:
                produced = build()
            except Exception:
                continue        # one unrenderable block shouldn't lose the rest
            text = produced if isinstance(produced, str) else _ansi(produced, width)
            if not text.endswith("\n"):
                text += "\n"
            sys.stdout.write(text)
        sys.stdout.flush()

    # ---- the frame ----

    def _rule_with_chip(self):
        chip = f" {self.session_label or _ui._DEFAULT_LABEL} "
        width = self._width()
        return [("class:frame.rule", "─" * max(width - len(chip) - 2, 0)),
                 ("class:frame.chip", chip), ("class:frame.rule", "──")]

    def _rule(self):
        return [("class:frame.rule", "─" * self._width())]

    def _width(self) -> int:
        try:
            return self._app.output.get_size().columns
        except Exception:
            return _ui.console.width

    def _status_line(self):
        glyph = _DOTS[int(__import__("time").monotonic() * 8) % len(_DOTS)]
        elapsed = _ui._format_elapsed(__import__("time").monotonic() - self._phase_start)
        return [("class:frame.spinner", f"{glyph} "),
                 ("class:frame.label", _ui._plain(self._status)),
                 ("class:frame.hint", f"  ({elapsed})")]

    def _approve_line(self):
        question = self._approval[0] if self._approval else ""
        return [("class:prompt.arrow", "❯ "), ("class:frame.label", question),
                 ("class:frame.hint", "   y / n")]

    def _hint(self):
        if self._choosing():
            return _ui._hint_segments(self.model,
                                       "↑↓ or click to choose  ·  or type your own  ·  "
                                       "⏎ submit  ·  ctrl+c dismiss")
        if self._asking():
            return _ui._hint_segments(self.model, self._ask[0] if self._ask else "")
        if self._busy():
            return _ui._hint_segments(self.model, _ui._HINT_BUSY)
        if self._approving():
            return _ui._hint_segments(self.model, "y approve  ·  n deny  ·  ctrl+c interrupt")
        return _ui._hint_segments(self.model, _HINT_TUI)

    def _build_layout(self) -> Layout:
        one = Dimension.exact(1)
        transcript_window = Window(self.transcript, wrap_lines=False,
                                    always_hide_cursor=True)
        input_row = Window(BufferControl(buffer=self._buffer), height=one, wrap_lines=False)
        prompt_row = VSplit([
            Window(FormattedTextControl(lambda: [("class:prompt.arrow", "❯ ")]),
                    height=one, width=Dimension.exact(2)),
            input_row,
        ], height=one)
        self._input_row = input_row

        return Layout(HSplit([
            transcript_window,
            ConditionalContainer(
                CompletionsMenu(max_height=8, scroll_offset=1),
                filter=Condition(lambda: self._idle() and bool(self._buffer.complete_state)),
            ),
            # A blank row on each side of the status line / picker, so neither
            # is jammed against the transcript above it or the frame below.
            ConditionalContainer(
                Window(height=one),
                filter=Condition(lambda: self._busy() or self._choosing()),
            ),
            ConditionalContainer(
                Window(FormattedTextControl(self._status_line), height=one,
                        always_hide_cursor=True),
                filter=Condition(self._busy),
            ),
            ConditionalContainer(
                Window(FormattedTextControl(self._options_lines), always_hide_cursor=True,
                        height=Dimension(min=1, max=10)),
                filter=Condition(self._choosing),
            ),
            ConditionalContainer(
                Window(height=one),
                filter=Condition(lambda: self._busy() or self._choosing()),
            ),
            Window(FormattedTextControl(self._rule_with_chip), height=one),
            ConditionalContainer(prompt_row, filter=Condition(self._accepts_typing)),
            ConditionalContainer(
                Window(FormattedTextControl(self._approve_line), height=one,
                        always_hide_cursor=True),
                filter=Condition(self._approving),
            ),
            Window(FormattedTextControl(self._rule), height=one),
            Window(FormattedTextControl(self._hint), height=one),
        ]), focused_element=input_row)

    # ---- keys ----

    def _build_keys(self) -> KeyBindings:
        keys = KeyBindings()

        @keys.add("enter", filter=Condition(lambda: self._idle()))
        def _submit(event):
            text = self._buffer.text
            self._buffer.reset()
            if text.strip():
                # Echo it into the transcript immediately: the input line is
                # cleared on submit, so otherwise the request would disappear
                # and the turn's output would have nothing above it.
                _ui.instruction(text)
                self._queue.put_nowait(text)

        @keys.add("enter", filter=Condition(self._asking))
        def _answer_question(event):
            """What you typed if you typed anything, otherwise the highlighted
            choice. Both are answers; neither is more correct than the other,
            which is why the list never blocks the keyboard."""
            typed = self._buffer.text.strip()
            self._buffer.reset()
            if typed:
                answer = typed
            elif self._options:
                answer = self._options[self._choice]
            else:
                answer = ""
            if answer:
                _ui.instruction(answer)
            self._resolve_ask(answer)

        @keys.add("up", filter=Condition(self._choosing))
        def _previous_choice(event):
            self._choice = (self._choice - 1) % len(self._options)
            self.invalidate()

        @keys.add("down", filter=Condition(self._choosing))
        def _next_choice(event):
            self._choice = (self._choice + 1) % len(self._options)
            self.invalidate()

        @keys.add("c-c")
        def _interrupt(event):
            if self._asking():
                self._buffer.reset()
                self._resolve_ask(None)      # dismissed; the tool reports that
            elif self._approving():
                self._answer(False)
            elif self._busy():
                if self.on_interrupt is not None:
                    self.on_interrupt()
            else:
                self._buffer.reset()

        @keys.add("c-d", filter=Condition(lambda: self._idle()))
        def _eof(event):
            if not self._buffer.text:
                self._exit_requested = True
                self._queue.put_nowait(None)

        @keys.add("y", filter=Condition(self._approving))
        @keys.add("Y", filter=Condition(self._approving))
        def _yes(event):
            self._answer(True)

        @keys.add("n", filter=Condition(self._approving))
        @keys.add("N", filter=Condition(self._approving))
        @keys.add("enter", filter=Condition(self._approving))
        def _no(event):
            self._answer(False)

        # Keyboard scrolling, for the same reasons a pager has it.
        @keys.add("pageup")
        def _page_up(event):
            self.transcript.scroll_by(-(self.transcript._last_height - 1),
                                       self.transcript._last_height)

        @keys.add("pagedown")
        def _page_down(event):
            self.transcript.scroll_by(self.transcript._last_height - 1,
                                       self.transcript._last_height)

        @keys.add("escape", "g")
        def _to_bottom(event):
            self.transcript.scroll_to_bottom()

        return keys

    # ---- running ----

    async def run(self):
        self._ticker = asyncio.ensure_future(self._tick())
        try:
            await self._app.run_async()
        finally:
            self._ticker.cancel()
            self._ticker = None

    async def _tick(self, interval: float = 0.1):
        """Repaint while busy, so the spinner turns and the counter climbs."""
        try:
            while True:
                if self._busy():
                    self.invalidate()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    def stop(self):
        self._exit_requested = True
        try:
            if self._app.is_running:
                self._app.exit()
        except Exception:
            pass

    async def next_instruction(self):
        """The next thing typed, or None when the session should end."""
        return await self._queue.get()

    # ---- modes ----

    def set_busy(self, label: str = "Thinking…"):
        self._status = label
        self._phase_start = __import__("time").monotonic()
        self._mode = "busy"
        self.invalidate()

    def set_label(self, label: str):
        self._status = label
        self._phase_start = __import__("time").monotonic()
        self.invalidate()

    def set_idle(self):
        self._mode = "idle"
        self.invalidate()

    async def ask_approval(self, question: str) -> bool:
        """Ask y/n in the frame, where every other prompt lives, instead of
        reading a line from a terminal this app is holding in raw mode."""
        previous = self._mode
        future = asyncio.get_running_loop().create_future()
        self._approval = (question, future)
        self._mode = "approve"
        self.invalidate()
        try:
            return await future
        finally:
            self._approval = None
            self._mode = previous
            self.invalidate()

    def _answer(self, verdict: bool):
        if self._approval and not self._approval[1].done():
            self._approval[1].set_result(verdict)

    async def ask_text(self, hint: str, options: list = None) -> str:
        """Answer a question the model asked (the ask_user tool), rather than
        start a new turn. Same input row — there is still only one place to
        type — but the line goes back to the tool, and when options were
        offered they are shown as a picker above it: arrow or click to choose,
        or ignore it and type something else entirely."""
        previous = self._mode
        future = asyncio.get_running_loop().create_future()
        self._ask = (hint, future)
        self._options = list(options or [])
        self._choice = 0
        self._mode = "ask"
        self.invalidate()
        try:
            return await future
        finally:
            self._ask = None
            self._options = []
            self._mode = previous
            self.invalidate()

    def _resolve_ask(self, text):
        if self._ask and not self._ask[1].done():
            self._ask[1].set_result(text)


_HINT_TUI = ("⏎ send  ·  / commands  ·  click ▸ to expand  ·  "
             "wheel/pgup scroll  ·  ctrl+d exit")
