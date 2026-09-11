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
import contextvars
import io
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text, split_lines
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

from . import clipboard

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

    def plain_text(self, width: int) -> str:
        """The whole transcript as text, styles dropped.

        Rows are already laid out as fragments, so this is the same wrapping
        the screen shows — including blocks that are scrolled off, which is
        the part a drag across the visible screen can never reach."""
        self._layout(width)
        # rstrip: a row is padded out to the full width on screen (Rich
        # centres a heading that way), and pasting those trailing spaces
        # somewhere else is never what was wanted.
        return "\n".join(fragment_list_to_text(fragments).rstrip()
                           for fragments, _ in self._rows)

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


class Pane:
    """One agent's screen.

    Main is pane 0; every subagent gets its own. A pane owns everything that
    belongs to *that* agent rather than to the window: its transcript, whether
    it's working, and any question of its own waiting to be answered. That
    last part is why approvals from a background subagent don't hijack the
    screen — they wait in its own tab until you go there."""

    def __init__(self, name: str, agent=None, kind: str = "main", depth: int = 0):
        self.name = name
        self.agent = agent
        self.kind = kind              # "main" | "subagent"
        self.depth = depth
        # The session this pane's agent is continuing. Each pane has its own,
        # which is what keeps two agents' histories apart in the database.
        self.session_id = None
        self.transcript = Transcript()

        self.busy = False
        self.status = ""
        self.phase_start = 0.0
        self.done = False             # a subagent that finished
        self.unseen = False           # output arrived while you were elsewhere
        # How its last turn ended: "" (never ran), "done", "interrupted" or
        # "error". What a subagent reports back to its parent depends on this,
        # so "it didn't finish" can say which.
        self.outcome = ""
        self.error = ""

        self.task = None              # the turn running now, if any
        self.pending: list = []       # (text, attachments) typed while busy
        # Images pasted at this pane, waiting for the prompt they belong to.
        # Each is {"mime": str, "data": bytes}; the nth is what "[Image #n]"
        # in the typed line refers to.
        self.attachments: list = []
        self.on_interrupt = None

        self.approval = None          # (question, Future)
        self.ask = None               # (hint, Future)
        self.options: list = []
        self.choice = 0

    # The frame shows exactly one of these for the focused pane.
    @property
    def mode(self) -> str:
        if self.ask is not None:
            return "ask"
        if self.approval is not None:
            return "approve"
        if self.busy:
            return "busy"
        return "idle"

    @property
    def needs_you(self) -> bool:
        return self.approval is not None or self.ask is not None


# The pane a turn's output belongs to. A ContextVar rather than an attribute
# because several agents run at once, each in its own task: a task inherits
# the context it was created in, so output finds its own pane without every
# renderer having to be told which one.
current_pane = contextvars.ContextVar("current_pane", default=None)


class _FocusedTranscript(UIControl):
    """The transcript window shows whichever pane has focus, so switching is
    a matter of which control this defers to — no rebuilding the layout."""

    def __init__(self, app):
        self._app = app

    def create_content(self, width, height):
        return self._app.pane.transcript.create_content(width, height)

    def mouse_handler(self, mouse_event):
        return self._app.pane.transcript.mouse_handler(mouse_event)

    def is_focusable(self) -> bool:
        return False

    def preferred_height(self, width, max_available_height, wrap_lines, get_line_prefix):
        return self._app.pane.transcript.preferred_height(
            width, max_available_height, wrap_lines, get_line_prefix)


class TuiApp:
    """The whole screen for an interactive session.

    A tab bar of panes, the focused pane's transcript (scroll with the wheel,
    click a ▸ line to open it), and the frame underneath. The frame always
    reflects the focused pane, and shows one thing at a time, so there is
    exactly one place to type:

      idle     ❯ and your text
      busy     what that agent is doing, with an elapsed counter
      approve  a y/n question about one of its tool calls
      ask      a question the model asked you, with optional choices

    The REPL doesn't read input directly: this app runs for the whole session
    and hands (pane, instruction) pairs over a queue, which is what lets main
    keep working while you read or type at a subagent.
    """

    def __init__(self, commands: dict, session_label: str = "", model: str = ""):
        if _ui is None:
            _bind_ui()
        self.session_label = session_label
        self.model = model
        self.commands = commands

        self.panes: list = [Pane(session_label or "main", kind="main")]
        self.focused = 0
        self._picking = False      # walking the agent tree with the keyboard
        self._pick_index = 0
        self._fold_task = None
        # While this is on the app stops asking the terminal for mouse
        # reports, which hands click-and-drag back to the terminal so its own
        # selection and copy work. See the ctrl+s binding.
        self._selecting = False

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
            # A filter, not True: the renderer turns mouse tracking off and on
            # as this changes, which is what makes selection mode possible.
            mouse_support=Condition(lambda: not self._selecting),
            erase_when_done=True,
        )

    # ---- panes ----

    @property
    def pane(self) -> Pane:
        return self.panes[min(self.focused, len(self.panes) - 1)]

    @property
    def main(self) -> Pane:
        return self.panes[0]

    def add_pane(self, name: str, agent=None, depth: int = 1) -> Pane:
        pane = Pane(name, agent=agent, kind="subagent", depth=depth)
        self.panes.append(pane)
        self.invalidate()
        return pane

    def close_pane(self, pane: Pane) -> bool:
        """Drop a finished subagent's tab. Main can't be closed."""
        if pane is self.main or pane not in self.panes:
            return False
        index = self.panes.index(pane)
        self.panes.remove(pane)
        self.focused = min(self.focused, len(self.panes) - 1)
        if index <= self.focused:
            self.focused = max(self.focused - 1, 0) if index < self.focused else self.focused
        self.invalidate()
        return True

    def focus(self, index: int):
        if 0 <= index < len(self.panes):
            self.focused = index
            self.pane.unseen = False
            self._buffer.reset()
            self.invalidate()

    def focus_pane(self, pane: Pane):
        if pane in self.panes:
            self.focus(self.panes.index(pane))

    def cycle(self, step: int):
        self.focus((self.focused + step) % len(self.panes))

    def _agent_tree(self):
        """The agents, as a tree under the prompt.

        ○ is working, ● (green) has finished and reported back, ! is waiting
        on you for an approval or an answer. Rows are clickable, and ctrl+↑↓
        walks them with Enter to switch. The tree only exists while there are
        subagents: once the last one finishes they fold into main's transcript
        and it disappears again (see _fold_finished)."""
        fragments = []
        for index, pane in enumerate(self.panes):
            last = index == len(self.panes) - 1
            branch = "  " if index == 0 else ("  └─ " if last else "  ├─ ")

            if pane.needs_you:
                dot, dot_style = "!", "class:agent.attention"
            elif pane.busy:
                dot, dot_style = "○", "class:agent.busy"
            elif pane.outcome in ("interrupted", "error"):
                # Finished, but not with an answer — worth telling apart from
                # a clean ●, since the parent was told as much.
                dot, dot_style = "✕", "class:agent.attention"
            elif pane.done:
                dot, dot_style = "●", "class:agent.done"
            else:
                dot, dot_style = "·", "class:agent"

            picked = self._picking and index == self._pick_index
            name_style = ("class:agent.picked" if picked else
                           "class:agent.active" if index == self.focused else "class:agent")
            note = ""
            if pane.busy and pane.status:
                note = f"  {_ui._plain(pane.status)} " \
                        f"({_ui._format_elapsed(time.monotonic() - pane.phase_start)})"
            elif pane.needs_you:
                note = "  waiting for you"
            elif pane.outcome == "interrupted":
                note = "  interrupted"
            elif pane.outcome == "error":
                note = f"  {pane.error}"[:40]
            elif pane.unseen:
                note = "  new output"

            def handler(event, target=index):
                if event.event_type == MouseEventType.MOUSE_UP:
                    self.focus(target)

            fragments.append(("class:agent", branch, handler))
            fragments.append((dot_style, dot + " ", handler))
            # Main is "main": it isn't one of the numbered subagents, and a
            # leading 0 beside the session name read like one.
            label = "main" if index == 0 else f"{index} {pane.name}"
            fragments.append((name_style, label, handler))
            fragments.append(("class:frame.hint", note, handler))
            if not last:
                fragments.append(("", "\n"))
        return fragments

    def invalidate(self):
        try:
            self._app.invalidate()
        except Exception:
            pass   # not running yet; the next render picks the state up anyway

    # ---- transcript ----

    def emit(self, collapsed, expanded=None, pane: Pane = None) -> Block:
        """Add one entry to the pane that produced it (see current_pane).
        Pass `expanded` to make it clickable."""
        target = pane or current_pane.get() or self.pane
        if target not in self.panes:
            target = self.pane
        block = target.transcript.add(Block(collapsed=collapsed, expanded=expanded))
        if target is not self.pane:
            target.unseen = True
        self.invalidate()
        return block

    def transcript_width(self) -> int:
        """Width the transcript renders at — what ui.py renders its blocks to."""
        return self._width()

    def dump(self):
        """Write the transcripts to the real terminal, in whatever open/closed
        state they were left in.

        A full-screen app hands the terminal back with its previous contents
        restored, which would otherwise mean the whole session vanishing from
        scrollback on exit. Written as raw ANSI rather than through Rich: the
        blocks are already rendered text, and re-printing them as markup would
        mangle the escapes they contain. Must be called while ui still has this
        app registered, since that is what the block closures render against."""
        width = self.transcript_width()
        for index, pane in enumerate(self.panes):
            if index:
                sys.stdout.write(_ansi(_ui.subagent_summary(pane.name,
                                                              len(pane.transcript.blocks)), width))
            for block in pane.transcript.blocks:
                build = block.expanded if (block.is_open and block.expanded) else block.collapsed
                try:
                    produced = build()
                except Exception:
                    continue    # one unrenderable block shouldn't lose the rest
                text = produced if isinstance(produced, str) else _ansi(produced, width)
                if not text.endswith("\n"):
                    text += "\n"
                sys.stdout.write(text)
        sys.stdout.flush()

    def copy_transcript(self, pane: Pane = None) -> tuple:
        """Put the focused pane's transcript on the system clipboard.

        Returns (copied, lines). Dragging across the screen can only ever
        reach the rows the screen is showing; this takes the transcript
        whole, in whatever open/closed state its blocks were left in."""
        target = pane or self.pane
        text = target.transcript.plain_text(self.transcript_width())
        if not text.strip():
            return False, 0
        return clipboard.copy_text(text), text.count("\n") + 1

    # ---- the frame ----

    def _rule_with_chip(self):
        chip = f" {self.pane.name or _ui._DEFAULT_LABEL} "
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
        """Spinner, what it's doing, and how long plus what it has spent —
        the tokens come from the server's usage blocks, so the count beside
        the spinner is the server's own."""
        pane = self.pane
        glyph = _DOTS[int(time.monotonic() * 8) % len(_DOTS)]
        detail = _ui._format_elapsed(time.monotonic() - pane.phase_start)
        tokens = getattr(pane.agent, "tokens", None) if pane.agent else None
        if tokens:
            counted = _ui.format_tokens(tokens.get("prompt", 0), tokens.get("completion", 0))
            if counted:
                detail += f" · {counted}"
        return [("class:frame.spinner", f"{glyph} "),
                 ("class:frame.label", _ui._plain(pane.status)),
                 ("class:frame.hint", f"  ({detail})")]

    def _approve_line(self):
        pane = self.pane
        question = pane.approval[0] if pane.approval else ""
        return [("class:prompt.arrow", "❯ "), ("class:frame.label", question),
                 ("class:frame.hint", "   y / n")]

    def _options_lines(self):
        """The choices, with the highlighted one marked. Each row carries its
        own mouse handler, so an option can be clicked as well as arrowed
        to — the transcript is clickable, and so is this."""
        pane = self.pane
        typing = bool(self._buffer.text.strip())
        fragments = []
        for i, option in enumerate(pane.options):
            selected = i == pane.choice and not typing
            marker = "▸ " if selected else "  "
            style = "class:choice.selected" if selected else "class:choice"

            def handler(event, index=i, target=pane):
                if event.event_type == MouseEventType.MOUSE_UP:
                    target.choice = index
                    self._buffer.reset()
                    self.invalidate()

            fragments.append((style, f" {marker}{i + 1}. {option}", handler))
            fragments.append(("", "\n"))
        if typing:
            fragments.append(("class:frame.hint", "   (using what you typed)"))
        return fragments

    # ---- modes, read off the focused pane ----

    @property
    def mode(self) -> str:
        return self.pane.mode

    def _idle(self) -> bool:
        return self.pane.mode == "idle"

    def _busy(self) -> bool:
        return self.pane.mode == "busy"

    def _approving(self) -> bool:
        return self.pane.mode == "approve"

    def _asking(self) -> bool:
        return self.pane.mode == "ask"

    def _choosing(self) -> bool:
        return self._asking() and bool(self.pane.options)

    def _accepts_typing(self) -> bool:
        """Idle and ask both read a line from the input row — the difference
        is only where the line goes."""
        return self._idle() or self._asking()

    def _hint(self):
        switch = "  ·  ctrl+←→ switch" if len(self.panes) > 1 else ""
        if self._selecting:
            return _ui._hint_segments(self.model, _HINT_SELECTING)
        if self._choosing():
            return _ui._hint_segments(self.model,
                                       "↑↓ or click to choose  ·  or type your own  ·  "
                                       "⏎ submit  ·  ctrl+c dismiss")
        if self._asking():
            return _ui._hint_segments(self.model, self.pane.ask[0] if self.pane.ask else "")
        if self._approving():
            return _ui._hint_segments(self.model, "y approve  ·  n deny  ·  ctrl+c interrupt")
        if self._busy():
            return _ui._hint_segments(self.model, _ui._HINT_BUSY + switch)
        return _ui._hint_segments(self.model, _HINT_TUI + switch)

    # ---- picking an agent from the tree ----

    def _pick_mode(self, on: bool):
        self._picking = on
        if on:
            self._pick_index = self.focused
        self.invalidate()

    def _move_pick(self, step: int):
        """First press enters pick mode *and* moves — pressing down once and
        having nothing move reads as the key not working."""
        if not self._picking:
            self._pick_mode(True)
        self._pick_index = (self._pick_index + step) % len(self.panes)
        self.invalidate()

    def _take_pick(self):
        self.focus(self._pick_index)
        self._pick_mode(False)

    # ---- folding finished subagents back into main ----

    def _subagents(self) -> list:
        return [p for p in self.panes if p.kind == "subagent"]

    def _all_subagents_finished(self) -> bool:
        subs = self._subagents()
        return bool(subs) and all(p.done and not p.busy and not p.needs_you for p in subs)

    def _schedule_fold(self, delay: float = 1.5):
        """Give the green dot a moment to be seen, then clear the tree."""
        if self._fold_task is not None and not self._fold_task.done():
            return
        try:
            self._fold_task = asyncio.ensure_future(self._fold_after(delay))
        except RuntimeError:
            pass    # no running loop (tests); folding is then explicit

    async def _fold_after(self, delay: float):
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if self._all_subagents_finished():
            self._fold_finished()

    def _fold_finished(self):
        """Move each finished subagent into main's transcript as one clickable
        block and drop its pane.

        The tree returns to just main, but nothing is lost: the block holds
        everything that agent did, its answer is already in main's
        conversation, and its session is still in the database."""
        width = self.transcript_width()
        for pane in self._subagents():
            if not pane.done:
                continue
            rendered = []
            for block in pane.transcript.blocks:
                build = block.expanded if (block.is_open and block.expanded) else block.collapsed
                try:
                    produced = build()
                except Exception:
                    continue
                rendered.append(produced if isinstance(produced, str)
                                 else _ansi(produced, width))
            body = "".join(rendered)
            head = _ui._rendered(_ui.subagent_summary._plain, pane.name,
                                  len(pane.transcript.blocks))
            self.emit(lambda head=head: head,
                       lambda head=head, body=body: head + body,
                       pane=self.main)
            self.panes.remove(pane)
        self.focused = min(self.focused, len(self.panes) - 1)
        self._picking = False
        self.invalidate()

    def _build_layout(self) -> Layout:
        one = Dimension.exact(1)
        input_row = Window(BufferControl(buffer=self._buffer), height=one, wrap_lines=False)
        prompt_row = VSplit([
            Window(FormattedTextControl(lambda: [("class:prompt.arrow", "❯ ")]),
                    height=one, width=Dimension.exact(2)),
            input_row,
        ], height=one)
        self._input_row = input_row

        return Layout(HSplit([
            Window(_FocusedTranscript(self), wrap_lines=False, always_hide_cursor=True),
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
            # The agent tree sits under the prompt, and only while there is
            # more than one agent to choose between.
            ConditionalContainer(
                Window(FormattedTextControl(self._agent_tree), always_hide_cursor=True,
                        height=Dimension(min=1, max=8)),
                filter=Condition(lambda: len(self.panes) > 1),
            ),
        ]), focused_element=input_row)

    # ---- keys ----

    def _build_keys(self) -> KeyBindings:
        keys = KeyBindings()

        def _navigating() -> bool:
            """An empty prompt with more than one agent means the arrows and
            Enter belong to the agent tree — you can see it below the input,
            so that is where they should go. The moment you type anything they
            go back to the input line.

            Busy counts: while an agent is working is exactly when you want to
            look at another one, and there is nothing else the arrows could
            mean then. Only a question in progress (a y/n approval, or a
            choice picker) keeps them, since those own Enter."""
            return (len(self.panes) > 1 and not self._buffer.text.strip()
                     and self.pane.mode in ("idle", "busy"))

        self._navigating = _navigating

        @keys.add("up", filter=Condition(_navigating))
        def _up_agent(event):
            self._move_pick(-1)

        @keys.add("down", filter=Condition(_navigating))
        def _down_agent(event):
            self._move_pick(1)

        @keys.add("enter", filter=Condition(lambda: _navigating() and self._picking))
        def _switch_agent(event):
            self._take_pick()

        @keys.add("enter", filter=Condition(lambda: self._idle() and not self._picking))
        def _submit(event):
            text = self._buffer.text
            self._buffer.reset()
            if text.strip():
                # Echo it into the transcript immediately: the input line is
                # cleared on submit, so otherwise the request would disappear
                # and the turn's output would have nothing above it.
                token = current_pane.set(self.pane)
                try:
                    _ui.instruction(text, images=len(self.pane.attachments))
                finally:
                    current_pane.reset(token)
                self._queue.put_nowait((self.pane, text))

        @keys.add("c-v", filter=Condition(self._accepts_typing))
        def _paste(event):
            """Paste — an image if the clipboard holds one, otherwise text.

            Cmd+V never reaches here: it is the terminal's own paste, and a
            terminal has no way to deliver image data. So this reads the
            clipboard itself, and when there is no image it does what the
            keypress would have done anyway."""
            grabbed = clipboard.grab_image()
            if grabbed is None:
                text = clipboard.clipboard_text()
                if text:
                    self._buffer.insert_text(text.replace("\n", " "))
                return
            mime, data = grabbed
            pane = self.pane
            pane.attachments.append({"mime": mime, "data": data})
            # The placeholder is what makes the attachment visible: you can
            # see that an image is going along, refer to it in the sentence
            # you are writing, and count how many you have pasted.
            self._buffer.insert_text(f"[Image #{len(pane.attachments)}]")

        @keys.add("enter", filter=Condition(lambda: self._asking()))
        def _answer_question(event):
            """What you typed if you typed anything, otherwise the highlighted
            choice. Both are answers; neither is more correct than the other,
            which is why the list never blocks the keyboard."""
            pane = self.pane
            typed = self._buffer.text.strip()
            self._buffer.reset()
            if typed:
                answer = typed
            elif pane.options:
                answer = pane.options[pane.choice]
            else:
                answer = ""
            if answer:
                token = current_pane.set(pane)
                try:
                    _ui.instruction(answer)
                finally:
                    current_pane.reset(token)
            self._resolve_ask(pane, answer)

        @keys.add("up", filter=Condition(self._choosing))
        def _previous_choice(event):
            pane = self.pane
            pane.choice = (pane.choice - 1) % len(pane.options)
            self.invalidate()

        @keys.add("down", filter=Condition(self._choosing))
        def _next_choice(event):
            pane = self.pane
            pane.choice = (pane.choice + 1) % len(pane.options)
            self.invalidate()

        @keys.add("c-c")
        def _interrupt(event):
            pane = self.pane
            if pane.ask is not None:
                self._buffer.reset()
                self._resolve_ask(pane, None)   # dismissed; the tool says so
            elif pane.approval is not None:
                self._answer(pane, False)
            elif pane.busy:
                if pane.on_interrupt is not None:
                    pane.on_interrupt()
            else:
                self._buffer.reset()

        @keys.add("c-d", filter=Condition(lambda: self._idle()))
        def _eof(event):
            if not self._buffer.text:
                self._exit_requested = True
                self._queue.put_nowait((None, None))

        @keys.add("y", filter=Condition(self._approving))
        @keys.add("Y", filter=Condition(self._approving))
        def _yes(event):
            self._answer(self.pane, True)

        @keys.add("n", filter=Condition(self._approving))
        @keys.add("N", filter=Condition(self._approving))
        @keys.add("enter", filter=Condition(self._approving))
        def _no(event):
            self._answer(self.pane, False)

        # Switching panes. Ctrl+arrows rather than tab, which belongs to
        # completion, or plain arrows, which belong to the choice picker.
        @keys.add("enter", filter=Condition(lambda: self._picking))
        def _take_pick(event):
            self._take_pick()

        @keys.add("escape", filter=Condition(lambda: self._picking))
        def _cancel_pick(event):
            self._pick_mode(False)

        @keys.add("c-up")
        def _pick_up(event):
            self._move_pick(-1)

        @keys.add("c-down")
        def _pick_down(event):
            self._move_pick(1)

        @keys.add("c-right")
        def _next_pane(event):
            self.cycle(1)

        @keys.add("c-left")
        def _previous_pane(event):
            self.cycle(-1)

        # Keyboard scrolling, for the same reasons a pager has it.
        @keys.add("pageup")
        def _page_up(event):
            height = self.pane.transcript._last_height
            self.pane.transcript.scroll_by(-(height - 1), height)

        @keys.add("pagedown")
        def _page_down(event):
            height = self.pane.transcript._last_height
            self.pane.transcript.scroll_by(height - 1, height)

        @keys.add("escape", "g")
        def _to_bottom(event):
            self.pane.transcript.scroll_to_bottom()

        @keys.add("c-s")
        def _toggle_selection(event):
            """Hand the mouse back to the terminal, so text can be selected.

            A full-screen app asks the terminal to report clicks and drags to
            it, and a terminal that is reporting them is no longer doing its
            own selection — which is why dragging across the transcript
            selects nothing. Turning the reports off restores it; the cost is
            that clicking ▸ to expand stops working until this is pressed
            again, hence a toggle rather than a mode you can end up stuck in
            without noticing. For the whole transcript, scrolled-off parts
            included, /copy is the better answer."""
            self._selecting = not self._selecting
            self.invalidate()

        return keys

    # ---- running ----

    async def run(self):
        if self._exit_requested:
            return      # stopped before it ever started; don't take the screen
        self._ticker = asyncio.ensure_future(self._tick())
        try:
            await self._app.run_async()
        finally:
            self._ticker.cancel()
            self._ticker = None

    async def _tick(self, interval: float = 0.1):
        """Repaint while anything is working, so spinners turn and counters
        climb — including a subagent's, which shows in its tab."""
        try:
            while True:
                if any(p.busy for p in self.panes):
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
        """The next (pane, text) typed, or (None, None) to end the session."""
        return await self._queue.get()

    # ---- status ----

    def set_busy(self, label: str = "Thinking…", pane: Pane = None):
        pane = pane or current_pane.get() or self.pane
        pane.busy = True
        pane.done = False
        pane.status = label
        pane.phase_start = time.monotonic()
        self.invalidate()

    def set_label(self, label: str, pane: Pane = None):
        pane = pane or current_pane.get() or self.pane
        pane.status = label
        pane.phase_start = time.monotonic()
        self.invalidate()

    def set_interrupt(self, callback, pane: Pane = None):
        """What Ctrl+C should cancel while this pane is working. Per pane, so
        interrupting the agent you're looking at doesn't stop another."""
        pane = pane or current_pane.get() or self.pane
        pane.on_interrupt = callback

    def set_idle(self, pane: Pane = None, done: bool = False):
        pane = pane or current_pane.get() or self.pane
        pane.busy = False
        pane.status = ""
        # done shows a green ● in the tree, so you can see it landed without
        # switching to it. Once every subagent is done the tree folds away.
        pane.done = done and pane.kind == "subagent"
        self.invalidate()
        if self._all_subagents_finished():
            self._schedule_fold()

    # ---- questions ----

    async def ask_approval(self, question: str, pane: Pane = None) -> bool:
        """Ask y/n in the frame, where every other prompt lives, instead of
        reading a line from a terminal this app is holding in raw mode.

        The question belongs to its own pane: a background subagent's write
        waits in its tab (flagged !) rather than seizing the screen from
        whatever you were reading."""
        pane = pane or current_pane.get() or self.pane
        future = asyncio.get_running_loop().create_future()
        pane.approval = (question, future)
        self.invalidate()
        try:
            return await future
        finally:
            pane.approval = None
            self.invalidate()

    def _answer(self, pane: Pane, verdict: bool):
        if pane.approval and not pane.approval[1].done():
            pane.approval[1].set_result(verdict)

    async def ask_text(self, hint: str, options: list = None, pane: Pane = None) -> str:
        """Answer a question the model asked (the ask_user tool), rather than
        start a new turn. Same input row — there is still only one place to
        type — but the line goes back to the tool, and when options were
        offered they are shown as a picker above it: arrow or click to choose,
        or ignore it and type something else entirely."""
        pane = pane or current_pane.get() or self.pane
        future = asyncio.get_running_loop().create_future()
        pane.ask = (hint, future)
        pane.options = list(options or [])
        pane.choice = 0
        self.invalidate()
        try:
            return await future
        finally:
            pane.ask = None
            pane.options = []
            self.invalidate()

    def _resolve_ask(self, pane: Pane, text):
        if pane.ask and not pane.ask[1].done():
            pane.ask[1].set_result(text)


_HINT_TUI = ("⏎ send  ·  / commands  ·  click ▸ to expand  ·  ctrl+s select text  ·  "
              "ctrl+d exit  ·  wheel/pgup scroll")
_HINT_SELECTING = ("selecting: drag to select, then copy as usual  ·  "
                    "ctrl+s back  ·  /copy takes the whole transcript")
