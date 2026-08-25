"""Rich-powered terminal presentation for the coding agent.

Nothing here affects agent logic — it's purely how things are shown. If rich
isn't installed, agent.py falls back to plain print() (see its import guard).
"""

import asyncio
import getpass
import json
import os
import re
import time
import zlib
from contextlib import asynccontextmanager, contextmanager

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    BufferControl, ConditionalContainer, FormattedTextControl, HSplit, Layout, VSplit, Window,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Confirm
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from . import __version__

console = Console()

# Claude Code's signature warm rust — the one accent color this UI is built
# around, instead of the previous mix of cyan/blue/magenta/yellow per
# widget. Used for the prompt arrow, mascot, spinner, and every "neutral,
# needs attention" panel border; green/red/yellow stay reserved for actual
# success/danger/caution semantics (diffs, done, errors).
ACCENT = "#D97757"

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _render_diff(diff_text: str) -> Text:
    """Render a unified diff (as produced by difflib.unified_diff) with a
    line-number gutter and red/green highlighting for removed/added lines —
    GitHub-style — instead of relying on pygments' diff-lexer coloring."""
    body = Text()
    old_no = new_no = None
    for line in diff_text.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            continue
        m = _HUNK_RE.match(line)
        if m:
            old_no, new_no = int(m.group(1)), int(m.group(2))
            body.append(f"{line}\n", style="dim cyan")
            continue
        if line.startswith("-"):
            gutter = old_no if old_no is not None else ""
            body.append(f"{gutter:>5} ", style="dim")
            body.append(f"{line}\n", style="bold red")
            if old_no is not None:
                old_no += 1
        elif line.startswith("+"):
            gutter = new_no if new_no is not None else ""
            body.append(f"{gutter:>5} ", style="dim")
            body.append(f"{line}\n", style="bold green")
            if new_no is not None:
                new_no += 1
        else:
            gutter = new_no if new_no is not None else ""
            body.append(f"{gutter:>5} ", style="dim")
            body.append(f"{line}\n")
            if old_no is not None:
                old_no += 1
            if new_no is not None:
                new_no += 1
    return body


def _diff_stats(diff_text: str) -> tuple:
    """Count added/removed lines in a unified diff, ignoring the --- /+++
    file-header lines."""
    added = removed = 0
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _search_summary(result: str) -> str:
    """search_files results can be dozens of matched lines — the step log
    should read as a count, not a code dump (the model still gets the full
    text; this only affects what's printed to the terminal)."""
    if result.strip() == "(no matches)":
        return "0 matches found"
    lines = [l for l in result.splitlines() if l and not l.startswith("...")]
    count = len(lines)
    stopped = "...[stopped at" in result
    suffix = " (stopped early — narrow your pattern or glob)" if stopped else ""
    return f"{count} match{'es' if count != 1 else ''} found{suffix}"


def _read_summary(result: str) -> str:
    """read_file's result is the raw file content — the step log should
    say how much was read, not echo the code (the model still gets the
    full text; this only affects what's printed to the terminal)."""
    lines = result.splitlines()
    return f"{len(lines)} line{'s' if len(lines) != 1 else ''} ({len(result)} chars)"

_STATUS_COLOR = {"done": "green", "running": "yellow", "max_steps": "yellow",
                 "interrupted": "yellow", "error": "red"}


def banner(task: str, model: str):
    console.print(Rule(f"[bold {ACCENT}]Coding Agent[/bold {ACCENT}]", style=ACCENT))
    console.print(f"[dim]model:[/dim] {model}")
    console.print(f"[dim]task:[/dim]  {task}\n")


# Widest the header box will ever draw. Narrow enough that widening a
# terminal never reflows it and shrinking has to be drastic to touch it.
_HEADER_WIDTH = 84

# Nothing boxed is drawn wider than this. Boxes that span the terminal are
# what a narrowing resize rewraps into broken box-drawing, since scrollback
# belongs to the terminal and can't be redrawn — a capped box only suffers
# below its own width.
_MAX_PANEL_WIDTH = 100


def _panel_width() -> int:
    return max(min(console.width, _MAX_PANEL_WIDTH), 20)

_MASCOT_ART = """
  ▄▄▄▄▄▄▄
 █  ●  ● █
 █    ▽   █
  ▀▀▄▄▄▀▀
  ╱ ╲╱ ╲╱
""".strip("\n")


def _greeting() -> str:
    """"Welcome back, <name>" from the OS user, or a plain hello if the
    platform won't say who's logged in."""
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    name = user.replace(".", " ").replace("_", " ").strip().title()
    return f"Welcome back, {name}!" if name else "Welcome!"


def _short_path(path: str) -> str:
    """~-collapsed absolute path, for the header's location line."""
    try:
        full = os.path.abspath(path)
    except Exception:
        return path
    home = os.path.expanduser("~")
    return f"~{full[len(home):]}" if full == home or full.startswith(home + os.sep) else full


# Left of the divider: who/where/what model. Right of it: two short
# reference sections. Both are static text about this app — the panel is
# built fresh on every redraw, so nothing here can go stale.
_START_TIPS = (
    "Type a task, press Enter",
    "/ opens the command menu",
    "Ctrl+C interrupts a turn",
)
_COMMAND_TIPS = (
    ("/mcp", "servers and their tools"),
    ("/model", "switch model mid-session"),
    ("/compact", "shrink a long history"),
    ("/btw", "a quick aside"),
)


def _reference_column() -> Table:
    col = Table.grid(padding=(0, 1))
    col.add_column(overflow="ellipsis", no_wrap=True)
    col.add_row(Text("Getting started", style=f"bold {ACCENT}"))
    for tip in _START_TIPS:
        col.add_row(Text(tip))
    col.add_row(Rule(style="dim"))
    col.add_row(Text("Handy commands", style=f"bold {ACCENT}"))
    for name, what in _COMMAND_TIPS:
        line = Text(f"{name:<9}", style="bold")
        line.append(what, style="dim")
        col.add_row(line)
    return col


def _identity_column(session_label: str, project_root: str) -> Table:
    col = Table.grid(padding=(0, 1))
    col.add_column(justify="center")
    col.add_row(Text(_greeting(), style=f"bold {ACCENT}"))
    col.add_row(Text(_MASCOT_ART, style=ACCENT))
    col.add_row(Text(""))
    col.add_row(Text(session_label, style="dim"))
    if project_root:
        col.add_row(Text(_short_path(project_root), style="dim"))
    return col


def header(session_label: str, project_root: str = ""):
    """Box-framed header, printed once when the REPL starts.

    Deliberately says nothing about the model. This box is scrollback the
    moment it's printed — a terminal can't go back and edit it — so a model
    named here would be wrong from the first /model switch onward, and the
    only fix would be printing a second copy of the whole box into the middle
    of the transcript. The live model name lives on the frame's hint line
    instead (see _hint_segments), which is rebuilt continuously."""
    # box.MINIMAL with no edge/header draws only the interior divider, so the
    # two columns are separated inside the panel's own border rather than
    # each carrying a box of its own.
    # Fixed width, not expand=True. The box is scrollback the moment it is
    # printed: a full-width one is rewrapped into broken box-drawing as soon
    # as the terminal narrows, while a compact one only suffers below its own
    # width. Shrinks on a narrow terminal, never grows past _HEADER_WIDTH.
    inner = max(min(console.width, _HEADER_WIDTH) - 4, 30)
    left = int(inner * 0.47)
    layout = Table(box=box.MINIMAL, show_header=False, show_edge=False,
                    show_lines=False, padding=(0, 2), border_style="dim",
                    width=inner)
    layout.add_column(width=left)
    layout.add_column(width=inner - left - 3)
    layout.add_row(Align.center(_identity_column(session_label, project_root)),
                    _reference_column())

    title = f"[bold {ACCENT}]Omni Coder[/bold {ACCENT}] [dim]v{__version__}[/dim]"
    console.print(Panel(layout, title=title, title_align="left",
                         border_style=ACCENT, expand=False, width=inner + 4))


class SlashCommandCompleter(Completer):
    """Pops a completion menu for '/' commands as soon as the input starts
    with '/' — the static REPL commands (/exit, /sessions, /compact, ...)
    plus, once connected, one entry per MCP prompt exposed by a connected
    server (formatted as "/server:prompt").

    `commands` is a {command_text: description} dict owned by the caller —
    this class only reads it at completion time, so the caller can mutate
    it in place (e.g. add prompt commands after the MCP client connects)
    without recreating the completer or the PromptSession."""

    def __init__(self, commands: dict):
        self.commands = commands

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in self.commands.items():
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), display_meta=desc)


# ---------------- the bottom frame ----------------
#
# The frame is the fixed furniture at the bottom of the terminal: a rule
# carrying the session name on the right, the input line (or, mid-turn, what
# the agent is doing), a closing rule, and a key hint. Everything else —
# model narration, tool calls, panels — scrolls in the region above it.
#
# It's owned by two different mechanisms that never run at once: while
# waiting for input, prompt_toolkit owns the bottom (its own prompt line plus
# a bottom_toolbar for the closing rule and hint, since Rich output would
# land above the input line rather than under it); while a turn runs,
# _BottomFrame's Rich Live owns it.

_FRAME_RULE = "#6c6c6c"
_FRAME_HINT = "#8a8a8a"
_HINT_IDLE = "⏎ send  ·  / commands  ·  ctrl+c clear  ·  ctrl+d exit"
_HINT_BUSY = "working…  ·  ctrl+c interrupts the turn"
_DEFAULT_LABEL = "omni-coder"

# noreverse/bg:default: prompt_toolkit renders a bottom toolbar as reverse
# video by default, which would paint a solid bar across the terminal.
def _hint_segments(model: str, keys: str) -> list:
    """The hint line: the live model name, then the key reminders.

    The model belongs here rather than only in the startup header — the header
    is scrollback and can't be updated in place, so a /model switch would
    leave it stale unless a second copy of the box were printed mid-transcript.
    This line is rebuilt on every prompt and every frame tick, so it's always
    the model actually in use."""
    segments = [("class:frame.hint", "  ")]
    if model:
        segments += [("class:frame.model", model), ("class:frame.hint", "  ·  ")]
    segments.append(("class:frame.hint", keys))
    return segments


_PROMPT_STYLE = Style.from_dict({
    "frame.rule": _FRAME_RULE,
    "frame.chip": f"bold reverse {ACCENT}",
    "frame.hint": _FRAME_HINT,
    "frame.model": f"bold {ACCENT}",
    "frame.spinner": f"bold {ACCENT}",
    "frame.label": f"bold {ACCENT}",
    "prompt.arrow": f"bold {ACCENT}",
    # bg:default everywhere except the selected row: prompt_toolkit's stock
    # menu paints a solid block, and since the menu is a full-width member of
    # the box's stack (not a float) that block reads as the whole terminal
    # going dark the moment you type "/".
    "completion-menu": "bg:default",
    "completion-menu.completion": "bg:default #d0d0d0",
    "completion-menu.completion.current": f"bold bg:{ACCENT} #1c1c1c",
    "completion-menu.meta.completion": "bg:default #8a8a8a",
    "completion-menu.meta.completion.current": f"bg:{ACCENT} #1c1c1c",
    "completion-menu.multi-column-meta": "bg:default #8a8a8a",
    "scrollbar.background": "bg:default",
    "scrollbar.button": f"bg:{_FRAME_RULE}",
})


_DOTS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_MARKUP_RE = re.compile(r"\[/?[a-zA-Z#][^\]]*\]")


def _plain(label: str) -> str:
    """Rich markup stripped: the busy line is drawn by prompt_toolkit, which
    would print "[bold yellow]…[/bold yellow]" literally."""
    return _MARKUP_RE.sub("", label or "")


class PromptBox:
    """The input box: a rule carrying the session chip, the ❯ line, a closing
    rule, and a key hint — four lines drawn directly beneath the transcript.

    A prompt_toolkit `bottom_toolbar` can't do this: it is pinned to the last
    row of the terminal, which leaves a block of blank lines between the input
    line and the closing rule on anything but a full screen of output. So the
    box is its own small (non-full-screen) Application whose container is
    exactly those four lines, rendered at the cursor — the transcript scrolls
    above it as content is printed.

    `commands` is the caller's live {command: description} dict, read at
    completion time (see SlashCommandCompleter), so the REPL can keep adding
    entries without rebuilding this."""

    def __init__(self, commands: dict):
        self.session_label = ""
        self.model = ""
        # Busy state: while a turn runs the same box shows what the agent is
        # doing instead of an input line. Rendered by prompt_toolkit rather
        # than a Rich Live so ONE renderer owns the bottom of the terminal at
        # all times — two of them sharing it (through patch_stdout, which is
        # itself prompt_toolkit's) is what left the spinner frozen and the
        # elapsed counter stuck on terminals where Rich's in-place repaint
        # didn't land.
        self._busy_label = "Thinking…"
        self._phase_start = 0.0
        self._busy_task = None
        self.on_interrupt = None   # set by the REPL: cancels the running turn
        self._buffer = Buffer(
            completer=SlashCommandCompleter(commands),
            complete_while_typing=True,
            history=InMemoryHistory(),
            multiline=False,
        )
        self._app = Application(
            layout=self._build_layout(),
            key_bindings=self._build_keys(),
            style=_PROMPT_STYLE,
            full_screen=False,
            erase_when_done=True,   # the box is furniture; the transcript keeps the text
            mouse_support=False,
        )
        self._busy_app = Application(
            layout=self._build_busy_layout(),
            key_bindings=self._build_busy_keys(),
            style=_PROMPT_STYLE,
            full_screen=False,
            erase_when_done=True,
            mouse_support=False,
        )

    # ---- layout ----

    def _width(self) -> int:
        try:
            return self._app.output.get_size().columns
        except Exception:
            return console.width

    def _rule_with_chip(self):
        chip = f" {self.session_label or _DEFAULT_LABEL} "
        dashes = max(self._width() - len(chip) - 2, 0)
        return [("class:frame.rule", "─" * dashes), ("class:frame.chip", chip),
                 ("class:frame.rule", "──")]

    def _rule(self):
        return [("class:frame.rule", "─" * self._width())]

    def _hint(self):
        return _hint_segments(self.model, _HINT_IDLE)

    # ---- busy mode ----

    @property
    def is_busy(self) -> bool:
        """Whether the busy frame is the box's current state.

        Deliberately "a busy task exists", not "the app is still running":
        without a terminal (tests, a piped stdin) the application exits
        immediately, and callers still need relabeling and stop_busy() to
        behave rather than silently diverging from the visible state."""
        return self._busy_task is not None

    def _busy_line(self):
        glyph = _DOTS[int(time.monotonic() * 8) % len(_DOTS)]
        elapsed = _format_elapsed(time.monotonic() - self._phase_start)
        return [("class:frame.spinner", f"{glyph} "),
                 ("class:frame.label", _plain(self._busy_label)),
                 ("class:frame.hint", f"  ({elapsed})")]

    def _busy_hint(self):
        return _hint_segments(self.model, _HINT_BUSY)

    def _build_busy_layout(self) -> Layout:
        one = Dimension.exact(1)
        # always_hide_cursor: this window has to be focusable (an Application
        # needs somewhere to put focus), and the terminal then parks its
        # cursor at the start of the line — painted as a solid block sitting
        # on top of the spinner glyph. Nothing is being typed here, so there
        # is no cursor to show.
        status = Window(FormattedTextControl(self._busy_line, focusable=True), height=one,
                         always_hide_cursor=True)
        return Layout(HSplit([
            status,
            Window(FormattedTextControl(self._rule_with_chip), height=one),
            Window(FormattedTextControl(self._busy_hint), height=one),
        ]), focused_element=status)

    def _build_busy_keys(self) -> KeyBindings:
        keys = KeyBindings()

        @keys.add("c-c")
        def _cancel(event):
            # prompt_toolkit holds the terminal in raw mode while this app
            # runs, so Ctrl+C arrives as a keystroke and never as SIGINT —
            # the REPL's signal handler would never see it.
            if self.on_interrupt is not None:
                self.on_interrupt()

        return keys

    def set_label(self, label: str):
        self._busy_label = label
        self._phase_start = time.monotonic()
        if self.is_busy:
            self._busy_app.invalidate()

    async def _run_busy(self):
        ticker = asyncio.ensure_future(self._tick_busy())
        try:
            await self._busy_app.run_async()
        finally:
            ticker.cancel()

    async def _tick_busy(self, interval: float = 0.1):
        """Repaint so the glyph advances and the counter climbs. invalidate()
        only schedules a redraw, so a failure here can't corrupt anything —
        and must not end the loop (that is exactly how a spinner stops
        spinning halfway through a turn)."""
        try:
            while True:
                try:
                    self._busy_app.invalidate()
                except Exception:
                    pass
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    async def start_busy(self, session_label: str = "", model: str = "",
                          label: str = "Thinking…"):
        if self.is_busy:
            return
        self.session_label = session_label or self.session_label
        self.model = model or self.model
        self._busy_label = label
        self._phase_start = time.monotonic()
        self._busy_task = asyncio.ensure_future(self._run_busy())
        # Let the app actually start before anything tries to stop it.
        for _ in range(20):
            if self._busy_app.is_running:
                break
            await asyncio.sleep(0.005)

    async def stop_busy(self):
        task, self._busy_task = self._busy_task, None
        if task is None:
            return
        if self._busy_app.is_running:
            self._busy_app.exit()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    def _build_layout(self) -> Layout:
        one = Dimension.exact(1)
        input_row = Window(BufferControl(buffer=self._buffer), height=one, wrap_lines=False)
        prompt_row = VSplit([
            Window(FormattedTextControl(lambda: [("class:prompt.arrow", "❯ ")]),
                    height=one, width=Dimension.exact(2)),
            input_row,
        ], height=one)
        # The completion menu is a plain member of the stack, shown only while
        # there are completions, rather than a Float: a float has to be given
        # room below the cursor, and the renderer buys that room by writing a
        # screenful of blank lines — which shoves the transcript up the moment
        # the box appears.
        menu = ConditionalContainer(
            CompletionsMenu(max_height=8, scroll_offset=1),
            filter=Condition(lambda: bool(self._buffer.complete_state)),
        )
        box = HSplit([
            menu,
            Window(FormattedTextControl(self._rule_with_chip), height=one),
            prompt_row,
            Window(FormattedTextControl(self._rule), height=one),
            Window(FormattedTextControl(self._hint), height=one),
        ])
        return Layout(box, focused_element=input_row)

    # ---- keys ----

    def _build_keys(self) -> KeyBindings:
        keys = KeyBindings()

        @keys.add("enter")
        def _accept(event):
            event.app.exit(result=self._buffer.text)

        @keys.add("c-c")
        def _interrupt(event):
            # Ctrl+C is "interrupt", never "quit": mid-turn it cancels the
            # turn (cli installs a SIGINT handler for that), and at the prompt
            # it clears whatever is half-typed. It used to raise
            # KeyboardInterrupt here, which the REPL treats as end-of-input —
            # so a stray Ctrl+C at the prompt dropped the whole session.
            # Leaving is /exit, /quit, or Ctrl+D on an empty line.
            self._buffer.reset()

        @keys.add("c-d")
        def _eof(event):
            if not self._buffer.text:
                event.app.exit(exception=EOFError, style="class:exiting")

        return keys

    # ---- driving it ----

    async def prompt(self, session_label: str = "", model: str = "") -> str:
        """Read one line. The box erases itself on submit; the instruction
        itself is re-printed into the transcript (plain, no "❯" — the arrow is
        part of the box, not of what you said) so the turn that follows has
        the request sitting above it."""
        self.session_label = session_label
        self.model = model
        self._buffer.reset()
        text = await self._app.run_async()
        if text.strip():
            console.print()
            console.print(Text(text, style="bold"))
        return text


async def prompt_task_async(box, session_label: str = "", model: str = "") -> str:
    """Read one instruction through the PromptBox, whose four-line frame sits
    directly under the transcript: everything printed while a turn runs
    (narration, tool calls, panels) scrolls above it, and the box itself is
    erased once the line is submitted. Must be called inside the caller's
    `patch_stdout()` context so Rich's output is redrawn above the box
    instead of corrupting it."""
    return await box.prompt(session_label, model)


def _chip(session_label: str) -> Text:
    """The session-name badge that sits at the right end of the frame's top
    rule. Rich-side twin of PromptBox._rule_with_chip, for the frame that's
    live while a turn runs."""
    return Text(f" {session_label or _DEFAULT_LABEL} ", style=f"bold reverse {ACCENT}")


def _frame_top(session_label: str) -> Rule:
    return Rule(_chip(session_label), align="right", style=_FRAME_RULE)


class _BottomFrame:
    """Holds the frame in place for a whole turn, via a single Rich Live.

    One Live, not one per phase: Rich allows only one live display at a
    time, and re-entering a new one per model call is what let the frame
    disappear and the transcript scroll freely between phases. So this owns
    the bottom for the turn's whole duration and `thinking()` merely
    relabels it (see .phase). Anything printed with console.print while it's
    open is drawn above the live region, which is exactly the layout we
    want: content on top, frame pinned below."""

    def __init__(self):
        self._live = None
        self._session = ""
        self._model = ""
        # One Spinner for the frame's whole life: Rich derives the animation
        # frame from how long *this instance* has been rendering, so building a
        # new one every tick pinned it to the first glyph — a spinner that
        # never spun.
        self._spinner = Spinner("dots", style=ACCENT)
        self._label = ""
        self._phase_start = 0.0
        self._task = None
        self._paused = False

    @property
    def is_open(self) -> bool:
        return self._live is not None

    def _render(self) -> Group:
        """Spinner first, then the frame — the running-turn status belongs
        directly above the prompt furniture (and below the transcript).

        No input line while a turn runs: the instruction the user typed is
        already sitting in the transcript above, so echoing an idle "❯" here
        would just be furniture claiming to accept input it can't take."""
        elapsed = _format_elapsed(time.monotonic() - self._phase_start)
        label = Text.from_markup(f"{self._label} [dim]({elapsed})[/dim]")
        # Every line of the frame must occupy exactly one row: Live repaints by
        # moving the cursor back over the previous render, so a line that wraps
        # at one width and not another throws that arithmetic off and leaves
        # the old copy stranded on screen.
        label.no_wrap = True
        label.overflow = "ellipsis"
        self._spinner.update(text=label)
        return Group(
            self._spinner,
            _frame_top(self._session),
            self._hint_line(),
        )

    def _hint_line(self) -> Text:
        line = Text("  ", no_wrap=True, overflow="ellipsis")
        if self._model:
            line.append(self._model, style=f"bold {ACCENT}")
            line.append("  ·  ", style=_FRAME_HINT)
        line.append(_HINT_BUSY, style=_FRAME_HINT)
        return line

    async def _tick(self, interval: float = 0.15):
        width = console.width
        try:
            while True:
                # Per-iteration, NOT around the loop: catching out here meant
                # one transient render error ended the ticker for the rest of
                # the turn, leaving a frozen glyph and a stuck counter. A bad
                # frame should cost one frame.
                try:
                    if not self._paused and self._live is not None:
                        if console.width != width:
                            # Resized. Live's incremental repaint is anchored
                            # to the geometry of the previous render, which no
                            # longer holds — without re-anchoring, every tick
                            # leaves another orphaned copy of the frame on
                            # screen (a staircase of rules and chips).
                            width = console.width
                            self._live.stop()
                            self._live.start(refresh=False)
                        self._live.update(self._render(), refresh=True)
                except Exception:
                    pass   # cosmetic furniture — never surface over real work
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    def open(self, session_label: str, model: str = "", label: str = "Thinking…"):
        if self._live is not None:
            return
        self._session = session_label
        self._model = model
        self._label = f"[bold {ACCENT}]{label}[/bold {ACCENT}]"
        self._phase_start = time.monotonic()
        # transient: erase the frame on close so it doesn't pile up in the
        # scrollback, one copy per turn.
        self._live = Live(self._render(), console=console, transient=True,
                           auto_refresh=False, redirect_stdout=False, redirect_stderr=False,
                           vertical_overflow="crop")
        self._live.__enter__()
        self._task = asyncio.ensure_future(self._tick())

    def close(self):
        if self._live is None:
            return
        if self._task is not None:
            self._task.cancel()
            self._task = None
        live, self._live = self._live, None
        live.__exit__(None, None, None)

    @contextmanager
    def pause(self):
        """Hand the terminal back for something that reads stdin (an approval
        prompt) or needs the cursor — a live region and a typed answer can't
        share the bottom of the screen."""
        if self._live is None:
            yield
            return
        self._paused = True
        self._live.stop()
        try:
            yield
        finally:
            self._live.start(refresh=True)
            self._paused = False

    def phase(self, label: str):
        """Relabel the frame for one phase of the turn (see _FramePhase)."""
        return _FramePhase(self, label)


class _FramePhase:
    """One phase of a turn, as seen by `thinking()`'s caller.

    Deliberately a class rather than a @contextmanager generator: callers
    hold the object *and* call .update() on it directly (agent._call_model
    relabels it on a retry), which a _GeneratorContextManager can't do. Same
    contract as _TickingSpinner: enter/exit plus .update()."""

    def __init__(self, frame, label: str):
        self._frame = frame
        self._label = label
        self._previous = None

    def __enter__(self):
        self._previous = (self._frame._label, self._frame._phase_start)
        self.update(self._label)
        return self

    def __exit__(self, *exc_info):
        if self._previous is not None:
            self._frame._label, self._frame._phase_start = self._previous
        return False

    def update(self, label: str):
        self._frame._label = label
        self._frame._phase_start = time.monotonic()


_frame = _BottomFrame()
_box = None   # the REPL's PromptBox, once it exists (see register_box)


def register_box(box):
    """Tell this module which PromptBox owns the bottom of the terminal, so
    the busy frame, `thinking()` and the approval prompt all go through the
    one renderer instead of a Rich Live competing with it."""
    global _box
    _box = box


@asynccontextmanager
async def turn_frame(session_label: str = "", model: str = "", on_interrupt=None):
    """Hold the frame at the bottom of the terminal for one turn.

    Through the registered PromptBox when there is one (the REPL): prompt_
    toolkit then owns that region for the whole session, redrawing it around
    every line the transcript prints. Falls back to the Rich Live frame
    otherwise, which is the one-shot `omni "task"` path where no box exists."""
    if _box is not None:
        _box.on_interrupt = on_interrupt
        await _box.start_busy(session_label, model)
        try:
            yield
        finally:
            await _box.stop_busy()
            _box.on_interrupt = None
        return
    _frame.open(session_label, model)
    try:
        yield
    finally:
        _frame.close()


def _format_elapsed(seconds: float) -> str:
    """Sub-minute durations stay decisecond-precise (e.g. "3.2s"); once a
    call runs a minute or longer, switch to whole-second m/s (or h/m/s past
    an hour) so a long wait reads as a duration, not a large decimal."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


class _TickingSpinner:
    """`with`-usable spinner (same contract as the plain rich Status this
    replaces: sync `__enter__`/`__exit__`, a `.update(label)` method) that
    keeps a live "(N.Ns)" elapsed-time suffix ticking on its own via a
    background asyncio task, instead of only reporting elapsed time once
    after the operation finishes (see elapsed_note/compacted for that).
    Must be entered from within a running event loop — true at every call
    site here, all inside `async def` functions.

    `.update(label)` keeps the exact prior contract: `label` is the full
    markup string to show (callers style it themselves, e.g. a differently
    colored retry message) — this class only appends the ticking suffix,
    it doesn't impose its own styling on updates."""

    def __init__(self, label: str, interval: float = 0.15):
        self._label = label
        self._interval = interval
        self._spinner = Spinner("dots", text=label, style=ACCENT)
        # auto_refresh=False plus redirect_stdout/stderr=False: drive every
        # redraw from this class's own tick loop alone — console.status()'s
        # default Live spawns its OWN background refresh thread and
        # redirects stdout independently, a second uncoordinated writer on
        # top of patch_stdout, which is already the one coordinating writes
        # against the REPL's prompt.
        self._live = Live(self._spinner, console=console, transient=True,
                           auto_refresh=False, redirect_stdout=False, redirect_stderr=False)
        self._task = None
        self._start = None

    def update(self, label: str):
        self._label = label

    async def _tick(self):
        try:
            while True:
                try:
                    elapsed = _format_elapsed(time.monotonic() - self._start)
                    self._spinner.update(text=f"{self._label} ({elapsed})")
                    self._live.refresh()
                except Exception:
                    pass  # one bad frame, not a dead ticker (see _BottomFrame._tick)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass

    def __enter__(self):
        self._start = time.monotonic()
        self._live.__enter__()
        self._task = asyncio.ensure_future(self._tick())
        return self

    def __exit__(self, *exc_info):
        if self._task is not None:
            self._task.cancel()
        return self._live.__exit__(*exc_info)


def thinking(label: str = "Thinking…"):
    """Spinner shown while waiting on a model call or tool execution, with a
    live elapsed-time counter.

    Inside a turn_frame (the interactive REPL) this relabels the pinned
    frame rather than opening a second live display — Rich permits only one,
    and the frame has to keep the bottom of the terminal for the whole turn.
    Outside one (a one-shot `omni "task"` run) it's a standalone spinner.
    Either way the returned object is a context manager with .update()."""
    if _box is not None and _box.is_busy:
        return _BoxPhase(_box, label)
    styled = f"[bold {ACCENT}]{label}[/bold {ACCENT}]"
    if _frame.is_open:
        return _frame.phase(styled)
    return _TickingSpinner(styled)


class _BoxPhase:
    """`thinking()`'s return value while the PromptBox is showing the busy
    frame. Same contract as _TickingSpinner and _FramePhase: a context
    manager that also answers .update() directly, which the retry path in
    agent._call_model calls on the object itself."""

    def __init__(self, box, label: str):
        self._box = box
        self._label = label
        self._previous = None

    def __enter__(self):
        self._previous = self._box._busy_label
        self._box.set_label(self._label)
        return self

    def __exit__(self, *exc_info):
        if self._previous is not None:
            self._box.set_label(self._previous)
        return False

    def update(self, label: str):
        self._box.set_label(label)


def elapsed_note(label: str, seconds: float):
    """Small dim line noting how long an operation took — printed once it
    finishes (after a `thinking()` spinner closes, or a step's tool calls
    are done executing), not a live-updating counter."""
    console.print(f"[dim]  {label} ({_format_elapsed(seconds)})[/dim]")


def intent_panel(intent, existing: dict):
    risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(intent.risk_level, "white")
    files_line = "none specified"
    if intent.target_files:
        parts = []
        for f in intent.target_files:
            tag = "[green]exists[/green]" if existing.get(f) else "[yellow]new[/yellow]"
            parts.append(f"{f} ({tag})")
        files_line = ", ".join(parts)
    constraints_line = "; ".join(intent.constraints) if intent.constraints else "none stated"
    confidence = "" if intent.confident else "\n[red]⚠ low confidence — parsing fell back to defaults[/red]"

    body = (
        f"[bold]type:[/bold] {intent.task_type}    "
        f"[bold]risk:[/bold] [{risk_color}]{intent.risk_level}[/{risk_color}]\n"
        f"[bold]summary:[/bold] {intent.summary}\n"
        f"[bold]files:[/bold] {files_line}\n"
        f"[bold]constraints:[/bold] {constraints_line}"
        f"{confidence}"
    )
    console.print(Panel(body, title="Parsed Intent", border_style=ACCENT, expand=False))


def high_risk_warning():
    console.print(Panel(
        "Approval required for ALL write/shell actions this run, even with --auto-approve.",
        title="⚠ High-risk task detected", border_style="red", expand=False,
    ))


_TOOL_EMOJI = {
    "read_file": "🔍", "write_file": "📝", "edit_file": "✏️",
    "list_dir": "📁", "glob_files": "🗂️", "search_files": "🔎",
    "run_shell": "💻",
    "git_diff": "📊", "git_status": "📋", "git_log": "📜", "git_show": "👁️",
    "git_branch": "🌿", "git_fetch": "📥", "git_add": "➕", "git_commit": "💾",
    "git_pull": "⬇️", "git_push": "⬆️",
    "save_memory": "🧠", "search_tools": "🧰",
    "list_resources": "📚", "read_resource": "📖",
}
_DEFAULT_TOOL_EMOJI = "🧩"  # fallback for an unnamespaced tool this map doesn't know

# Custom MCP server tools are exposed as "<server_name>__<tool_name>" (see
# mcp_client.py's list_llm_tools). Rather than one blanket icon for every
# custom tool, pick one deterministically per *server* name (a stable hash,
# not Python's randomized-per-process hash()) — so all of one server's tools
# share an icon, different servers get different ones, and it's the same
# icon across runs, not just within one session.
_SERVER_EMOJI_PALETTE = ["🔧", "🔌", "🛰️", "📡", "🧪", "🎛️", "🧬", "🪄"]


def _emoji_for(name: str) -> str:
    if name in _TOOL_EMOJI:
        return _TOOL_EMOJI[name]
    if "__" in name:
        server = name.split("__", 1)[0]
        return _SERVER_EMOJI_PALETTE[zlib.crc32(server.encode()) % len(_SERVER_EMOJI_PALETTE)]
    return _DEFAULT_TOOL_EMOJI


def _format_args(args: dict, max_len: int = 60) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            shown = v if len(v) <= max_len else v[:max_len] + "…"
            display = json.dumps(shown, ensure_ascii=False)
        else:
            display = repr(v)
        parts.append(f"{k}={display}")
    return ", ".join(parts)


def _call_str(name: str, args) -> str:
    emoji = _emoji_for(name)
    if args is None:
        return f"{emoji} {name}(<malformed arguments>)"
    return f"{emoji} [bold]{name}[/bold]({_format_args(args)})"


async def request_approval(name: str, args: dict, client) -> bool:
    """Show a rich preview (diff / content / command) and ask for confirmation.
    `client` is an MCPToolClient — previews go through the same MCP server
    the real tool calls do, just via the read-only _preview_* tools.

    The frame steps aside for the duration: the answer is typed at the bottom
    of the screen, which is the one place a live region can't share — and the
    busy app holds the terminal in raw mode, where a y/n prompt can't read a
    line at all."""
    if _box is not None and _box.is_busy:
        await _box.stop_busy()
        try:
            return await _request_approval(name, args, client)
        finally:
            await _box.start_busy()
    with _frame.pause():
        return await _request_approval(name, args, client)


async def _request_approval(name: str, args: dict, client) -> bool:
    console.print()   # the preview is its own block, like every other one
    emoji = _emoji_for(name)
    if name == "edit_file":
        path = args.get("path", "")
        ok, preview = await client.preview_edit(path, args.get("old_str", ""), args.get("new_str", ""))
        if not ok:
            console.print(Panel(f"[red]{preview}[/red]", title=f"{emoji} edit_file: {path}",
                                 border_style="red", width=_panel_width()))
            return False
        added, removed = _diff_stats(preview)
        title = f"{emoji} edit_file: {path}  [green]+{added}[/green] [red]-{removed}[/red]"
        console.print(Panel(_render_diff(preview), title=title, border_style="yellow",
                             width=_panel_width()))
    elif name == "write_file":
        path = args.get("path", "")
        is_new, preview = await client.preview_write(path, args.get("content", ""), args.get("overwrite", False))
        added, removed = _diff_stats(preview)
        label = "new" if is_new else "overwrite"
        title = f"{emoji} write_file ({label}): {path}  [green]+{added}[/green] [red]-{removed}[/red]"
        console.print(Panel(_render_diff(preview), title=title, width=_panel_width(),
                             border_style="green" if is_new else "yellow"))
    elif name == "run_shell":
        cmd = args.get("command", "")
        console.print(Panel(Syntax(cmd, "bash", theme="ansi_dark"),
                             title=f"{emoji} run_shell", border_style=ACCENT,
                             width=_panel_width()))
    else:
        console.print(Panel(json.dumps(args, indent=2), title=f"{emoji} {name}",
                             border_style="white", width=_panel_width()))

    return Confirm.ask("[bold]Proceed?[/bold]", default=False)


_DIFF_TOOLS = {"edit_file", "write_file"}


def _result_line(name: str, args, result: str, ok: bool, duration: float = None) -> tuple:
    """Returns (summary_line, diff_body_or_None) for one tool call's result."""
    icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
    path = args.get("path", "") if isinstance(args, dict) else ""
    diff_body = None
    if ok and name in _DIFF_TOOLS and "\n" in result:
        diff_body = result.split("\n", 1)[1]

    if ok and name == "search_files":
        summary = _search_summary(result)
    elif ok and name == "read_file":
        summary = _read_summary(result)
    elif diff_body is not None:
        added, removed = _diff_stats(diff_body)
        summary = f"{path}  +{added} -{removed}"
    else:
        summary = result.splitlines()[0] if result else ""
    time_suffix = f" [dim]· {_format_elapsed(duration)}[/dim]" if duration is not None else ""
    return f"{icon} {summary[:160]}{time_suffix}", diff_body


def step_display(calls: list):
    """`calls` is an ordered list of dicts with name, args (None if the
    model sent malformed JSON), result, ok, duration (seconds, absent for
    calls that never executed) — every tool call the model made this turn,
    already executed. Each call prints its call line then an indented ⎿
    result line, flat and sequential regardless of how many ran in
    parallel — no step numbering or boxed grouping."""
    for index, c in enumerate(calls):
        console.print()   # one blank line above every call, so a step with
                           # several of them reads as separate blocks
        console.print(f"[{ACCENT}]{_call_str(c['name'], c['args'])}[/{ACCENT}]")
        line, diff_body = _result_line(c["name"], c["args"], str(c["result"]), c["ok"], c.get("duration"))
        console.print(f"  [dim]⎿[/dim]  {line}")
        if diff_body is not None:
            console.print(Padding(_render_diff(diff_body), (0, 0, 0, 5)))


def assistant_message(text: str):
    """What the model said on a turn that also called tools.

    Printed above that step's tool lines, so the transcript reads as the
    reasoning followed by the actions it led to — previously this text was
    dropped on the floor and only the tool calls showed up. The last turn's
    text (the one with no tool calls) goes through final_result instead."""
    body = Table.grid(padding=(0, 1))
    body.add_column(width=1, justify="left", style=ACCENT, no_wrap=True)
    body.add_column(overflow="fold")
    body.add_row("●", Markdown(text))
    console.print()
    console.print(body)


def final_result(text: str):
    """The turn's answer: rendered as Markdown, with no rule and no border.

    A panel around it means every copy-paste drags box-drawing characters
    along, and the answer is the one thing people copy. An empty answer still
    gets called out — a blank space there is indistinguishable from "the
    terminal isn't showing the response"."""
    console.print()
    if not (text or "").strip():
        console.print(Text("The model returned an empty response — no text and no tool "
                            "call. Check agent_run.log for the raw reply.", style="dim yellow"))
        console.print()
        return
    console.print(Markdown(text))
    console.print()   # air between the answer and the input box below it


def history_panel(messages: list):
    """Show the prior conversation being resumed, rendered the same way it
    looked the first time around — assistant replies as Markdown panels,
    same as final_result() — so it's visibly clear context carried over
    instead of silently feeding the model in the background."""
    visible = [m for m in messages if m.get("role") != "system"]
    console.print(Rule(f"[bold {ACCENT}]Resumed history — {len(visible)} messages[/bold {ACCENT}]"))

    for m in visible:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "user":
            console.print(f"\n[bold {ACCENT}]❯[/bold {ACCENT}] {content}")
        elif role == "assistant" and m.get("tool_calls"):
            calls = ", ".join(f"{c['function']['name']}(…)" for c in m["tool_calls"])
            console.print(f"[dim]  → called {calls}[/dim]")
        elif role == "assistant":
            if content:
                console.print(Panel(Markdown(content), border_style=ACCENT, expand=False,
                                     width=_panel_width()))
        elif role == "tool":
            summary = content.splitlines()[0] if content else ""
            console.print(f"  [dim]✓ {summary[:150]}[/dim]")

    console.print(Rule(style="dim"))


def sessions_table(sessions: list):
    table = Table(title="Saved Sessions", expand=False)
    table.add_column("id", style=f"bold {ACCENT}")
    table.add_column("name", style="bold")
    table.add_column("status")
    table.add_column("updated", style="dim")
    table.add_column("model", style="dim")
    table.add_column("task")

    for s in sessions:
        color = _STATUS_COLOR.get(s["status"], "white")
        task = s["task"] if len(s["task"]) <= 60 else s["task"][:60] + "…"
        table.add_row(s["id"], s.get("name") or "-", f"[{color}]{s['status']}[/{color}]",
                      s["updated_at"], s["model"], task)

    console.print(table)


def resources_table(resources: dict):
    """Table for the /resources REPL command — one row per resource
    published by a connected MCP server. `resources` is
    MCPToolClient.list_resources()'s {uri: {...}} mapping."""
    if not resources:
        console.print("[dim]No resources published by the connected MCP servers.[/dim]")
        return

    table = Table(title="MCP Resources", expand=False)
    table.add_column("uri", style=f"bold {ACCENT}")
    table.add_column("server", style="bold")
    table.add_column("type", style="dim")
    table.add_column("description")

    for uri, info in resources.items():
        kind = "template" if info.get("template") else (info.get("mime_type") or "-")
        detail = info.get("description") or info.get("name") or ""
        if info.get("shadowed_by"):
            detail = f"{detail} [dim](also on: {', '.join(info['shadowed_by'])})[/dim]".strip()
        table.add_row(uri, info.get("server", ""), kind, detail)

    console.print(table)
    console.print("[dim]Read one with /resources <uri>[/dim]")


def resource_content(uri: str, content: str):
    """A single resource's contents, for `/resources <uri>`."""
    console.print(Panel(content or "(empty)", title=f"📖 {uri}", border_style=ACCENT, expand=False))


def server_tools_table(server: str, tools: list):
    """Table for `/mcp tools <name>` — one row per tool that server exposes.
    `tools` is MCPToolClient.server_tools()'s list. The name column shows the
    name the model calls (namespaced for custom servers); a "status" column
    only appears when there's something worth saying, so the common case
    stays a plain two-column list."""
    if not tools:
        console.print(f"[dim]{server} exposes no tools.[/dim]")
        return

    def status(t):
        if t["internal"]:
            return "[dim]internal[/dim]"
        if t["deferred"]:
            return "[yellow]deferred[/yellow]"
        if t["revealed"]:
            return f"[{ACCENT}]revealed[/{ACCENT}]"
        return ""

    interesting = any(status(t) for t in tools)
    table = Table(title=f"Tools on {server}", expand=False)
    table.add_column("tool", style=f"bold {ACCENT}")
    if interesting:
        table.add_column("status")
    table.add_column("description")

    for t in tools:
        desc = t["description"].splitlines()[0] if t["description"] else ""
        row = [f"{_emoji_for(t['name'])} {t['name']}"]
        if interesting:
            row.append(status(t))
        row.append(desc[:80])
        table.add_row(*row)

    console.print(table)
    shown = sum(1 for t in tools if not t["internal"] and not t["deferred"])
    note = f"{shown} of {len(tools)} callable by the model right now"
    if any(t["deferred"] for t in tools):
        note += " — deferred ones load via search_tools"
    console.print(f"[dim]{note}[/dim]")


def mcp_status(entries: list):
    """Table for the /mcp REPL command — one row per configured MCP server
    (built-in + custom), whether or not it actually connected. `entries` is
    MCPToolClient.server_status()'s output."""
    table = Table(title="MCP Servers", expand=False)
    table.add_column("")
    table.add_column("server", style="bold")
    table.add_column("status")
    table.add_column("tools", justify="right")
    table.add_column("target / error", style="dim")

    for e in entries:
        name = e["name"]
        if e["deferred"]:
            name += " [dim](defer)[/dim]"
        if e["connected"]:
            icon = "[green]✅[/green]"
            status = f"[green]connected {_format_elapsed(e['connected_for'])}[/green]"
            tools = str(e["tool_count"])
            detail = e["target"]
        else:
            icon = "[red]❌[/red]"
            status = "[red]failed[/red]"
            tools = "-"
            detail = f"[red]{e['error']}[/red]" if e["error"] else "-"
        table.add_row(icon, name, status, tools, detail)

    console.print(table)


def model_switched(model: str):
    """One line confirming a /model switch — the counterpart to not redrawing
    the header (which is why the frame's hint line carries the live name)."""
    line = Text("model → ", style="dim")
    line.append(model, style=f"bold {ACCENT}")
    console.print(line)


def interrupted():
    console.print(
        "\n[yellow]⏹ Interrupted — back at the prompt. Progress up to the last "
        "completed step was saved; keep chatting or ask the agent to continue.[/yellow]"
    )


def compacted(num_messages: int, summary_len: int, elapsed: float):
    console.print(
        f"[dim]⚙ compacted {num_messages} earlier messages into a "
        f"{summary_len}-char summary ({_format_elapsed(elapsed)}) to stay within the context budget[/dim]"
    )


def btw_answer(question: str, answer: str):
    """A /btw side question's answer, boxed distinctly ("💬 /btw" title)
    rather than looking like part of the surrounding task output."""
    console.print(Panel(Markdown(answer), title=f"💬 /btw: {question}", border_style=ACCENT, expand=False))


def warning(text: str):
    console.print(f"[yellow]⚠ {text}[/yellow]")


def error(text: str):
    console.print(f"[red]✗ {text}[/red]")
