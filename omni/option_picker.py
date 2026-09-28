"""The choices an `ask_user` question offers, when there is no full-screen
frame to put them in.

An interactive session draws them inside its own frame (tui.py): one row
marked, arrows to move, Enter to take it. A one-shot run (`omni "…"`) never
builds that frame, and the same question came down to a numbered list and a
bare `input()` — nothing marked, the arrows inert, Enter on an empty line
answering nothing at all. The question looked identical in both places and
behaved differently in one of them, which is the part worth fixing: a list
of choices that shows no cursor is a list you have to read twice to learn
it isn't a menu.

So this is that picker with no frame around it — the same marker, the same
keys, and the same rule that what you type beats what is highlighted —
printed under the question and erased once it is answered. What was chosen
is echoed by the caller, so the transcript keeps a record of an answer the
picker itself took off the screen.

Everything above `OptionPicker.run` is pure: the rows, the marker and the
answer a given state resolves to are all decided without a terminal
attached, which is what makes them testable.
"""

import sys

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    BufferControl, FormattedTextControl, HSplit, Layout, VSplit, Window,
)
from prompt_toolkit.layout.dimension import Dimension

from .session_picker import _Placeholder, scroll_window

# How many choices are on screen at once. A question with more than this
# many answers scrolls rather than pushing the question it belongs to off
# the top of the terminal.
VISIBLE_ROWS = 8


def available() -> bool:
    """Whether there is a terminal to draw the picker on.

    A piped or redirected run has no arrow keys to press and no screen to
    erase; prompt_toolkit would raise or hang rather than quietly degrade,
    so the caller keeps the numbered-list fallback for those.

    isatty() rather than anything platform-specific: it is the one answer
    every OS gives the same way, and what counts as a terminal underneath —
    a tty on macOS and Linux, a console screen buffer on Windows — is
    prompt_toolkit's problem, settled by `prepare()`."""
    for stream in (sys.stdin, sys.stdout):
        try:
            if not stream.isatty():
                return False
        except (AttributeError, ValueError):
            return False
    return True


class OptionPicker:
    """The list of choices itself. Built without a terminal so the selection
    logic can be exercised directly; `run()` is the only part that needs one."""

    def __init__(self, options: list, rows: int = VISIBLE_ROWS):
        self.options = list(options or [])
        self.rows = rows
        self.index = 0
        self.typed = ""
        self._app = None

    # ---- state ----

    @property
    def typing(self) -> bool:
        """Something typed takes the highlight off the list: the two are
        alternative answers, and showing a marked row beside text you are
        still writing would claim Enter is going to send the row."""
        return bool(self.typed.strip())

    @property
    def selected(self):
        return self.options[self.index] if 0 <= self.index < len(self.options) else None

    def move(self, delta: int):
        """Wraps, the way the in-frame picker does — with three choices all
        visible at once, getting from the last back to the first should not
        take four keystrokes."""
        if self.options:
            self.index = (self.index + delta) % len(self.options)

    def write(self, text: str):
        self.typed = text

    def answer(self) -> str:
        """What Enter sends: what you typed if you typed anything, otherwise
        the highlighted choice. Neither is more correct than the other, which
        is why the list never blocks the keyboard. A bare number is left as
        text — `ui.ask_user` maps it onto the option it names, so typing 2 and
        arrowing to the second row mean the same thing."""
        if self.typing:
            return self.typed.strip()
        return self.selected or ""

    # ---- rendering ----

    def row_fragments(self) -> list:
        """The choices, with the highlighted one marked. ❯ rather than a
        background block: the row sits directly under a question drawn by
        Rich, and a painted bar there reads as a different application."""
        total = len(self.options)
        if not total:
            return []
        first, last = scroll_window(self.index, total, self.rows)
        out = []
        for position in range(first, last):
            chosen = position == self.index and not self.typing
            if chosen:
                marker = ("class:picker.marker", "  ❯ ")
            elif position == first and first > 0:
                marker = ("class:picker.scroll", "  ↑ ")
            elif position == last - 1 and last < total:
                marker = ("class:picker.scroll", "  ↓ ")
            else:
                marker = ("class:picker.scroll", "    ")
            style = "class:picker.name.selected" if chosen else "class:picker.name"
            out += [marker, (style, f"{position + 1}. {self.options[position]}"), ("", "\n")]
        if self.typing:
            # The marker is gone at this point, so without a word here the
            # list just looks switched off.
            out.append(("class:picker.hint", "    (using what you typed)"))
        return out

    def hint_fragments(self) -> list:
        return [("class:picker.hint",
                  "  ↑↓ to move  ·  Enter to choose  ·  "
                  "or type your own answer  ·  Ctrl+C to dismiss")]

    # ---- the terminal part ----

    def _build(self) -> Application:
        from . import ui

        one = Dimension.exact(1)
        buffer = Buffer(multiline=False,
                         on_text_changed=lambda b: (self.write(b.text), self.invalidate()))
        entry = VSplit([
            Window(FormattedTextControl([("class:prompt.arrow", "  ❯ ")]),
                    width=Dimension.exact(4), height=one),
            Window(BufferControl(buffer=buffer,
                                  input_processors=[_Placeholder("or type your own answer…")]),
                    height=one, wrap_lines=False),
        ], height=one)

        layout = Layout(HSplit([
            Window(FormattedTextControl(self.row_fragments), always_hide_cursor=True,
                    height=Dimension(min=1, max=self.rows + 1)),
            Window(height=one),
            entry,
            Window(FormattedTextControl(self.hint_fragments), wrap_lines=True,
                    always_hide_cursor=True, height=Dimension(min=1, max=2)),
        ]), focused_element=entry)

        keys = KeyBindings()

        @keys.add("up")
        def _previous(event):
            self.move(-1)

        @keys.add("down")
        def _next(event):
            self.move(1)

        @keys.add("enter", filter=Condition(lambda: bool(self.answer())))
        def _accept(event):
            event.app.exit(result=self.answer())

        @keys.add("escape", eager=True)
        @keys.add("c-c")
        @keys.add("c-d")
        def _dismiss(event):
            event.app.exit(result=None)

        return Application(layout=layout, key_bindings=keys, style=ui._PROMPT_STYLE,
                            full_screen=False, erase_when_done=True)

    def invalidate(self):
        if self._app is not None:
            self._app.invalidate()

    def prepare(self) -> bool:
        """Build the app now, and say whether it could be built at all.

        Asked before the question is printed, because that is what decides
        how the choices are shown — a picker that turns out to be undrawable
        after the question has gone out leaves a list nobody can see and a
        hint about arrow keys that do nothing.

        Construction is where an unusable terminal actually says so:
        prompt_toolkit resolves its output here, which is the step that
        raises on a Windows console it cannot attach to (Cygwin, mintty, a
        stray TERM) and on a POSIX terminal it cannot put into raw mode.
        isatty() cannot see any of that, so it is checked by trying."""
        try:
            self._app = self._build()
        except Exception:
            self._app = None
            return False
        return True

    async def run(self):
        """Show the choices and return the answer, or None if it was
        dismissed."""
        if self._app is None and not self.prepare():
            raise RuntimeError("no terminal to draw the choices on")
        try:
            return await self._app.run_async()
        finally:
            self._app = None


async def pick_option(options: list):
    return await OptionPicker(options).run()
