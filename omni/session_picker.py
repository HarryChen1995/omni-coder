"""The list `--resume` opens when it is given no session to resume.

`--resume <id>` assumes you know the id, which you only do if you just ran
`--list-sessions` and copied one out of a table. Bare `--resume` is the
common case — "put me back in the one I was in" — so it opens the sessions
themselves: type to narrow, arrows to move, Enter to go.

The list opens on where you are actually standing — this project, this
branch — because that is where the session you are reaching for almost
always is, and a session from a branch you left three weeks ago is noise in
front of it. Ctrl+A widens it to every project, Ctrl+T to every branch.

Ctrl+T rather than the Ctrl+B this list's cousins use, because everything
nearer to hand is already spoken for and a key that means two things is
worse than an unfamiliar one: Ctrl+V and Ctrl+B both paste an image (Ctrl+B
because Windows Terminal eats Ctrl+V), Ctrl+S is selection mode, Ctrl+C
interrupts, Ctrl+D leaves, and the Ctrl+arrows walk the agent tree. See
RESERVED_KEYS.

Everything above `SessionPicker.run` is pure: the filtering, the ordering,
the row text and the scroll window are all decided without a terminal
attached, which is what makes them testable.
"""

import os
from datetime import datetime, timezone

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    BufferControl, ConditionalContainer, FormattedTextControl, HSplit, Layout, VSplit, Window,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import Processor, Transformation

# How many sessions are on screen at once. Six pairs of lines plus the search
# box, the heading and the hints is about as much as fits under a prompt
# without the list becoming the whole terminal.
VISIBLE_ROWS = 6

# Chords the session already uses for something else, which anything added
# here has to stay off. Not enforced by prompt_toolkit — the picker is its
# own Application, so it *could* rebind any of them — but a key that means
# paste everywhere else and "widen the list" here is worse than a key nobody
# has a habit for yet.
RESERVED_KEYS = frozenset({
    "c-v",                                    # paste an image
    "c-b",                                    # paste an image (Windows Terminal eats c-v)
    "c-s",                                    # selection mode, for the terminal's own copy
    "c-up", "c-down", "c-left", "c-right",    # walk the agent tree
})


def relative_age(when: str, now: datetime = None) -> str:
    """"2 days ago" for an ISO timestamp. Falls back to the raw string for
    anything unparseable, which is better than hiding the row's only clue
    about when it ran."""
    try:
        moment = datetime.fromisoformat(when)
    except (TypeError, ValueError):
        return when or ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    seconds = max((now - moment).total_seconds(), 0)
    for size, unit in ((60, "second"), (3600, "minute"), (86400, "hour"),
                        (604800, "day"), (2592000, "week")):
        if seconds < size:
            break
    else:
        size, unit = 31536000, "month"
    step = {"second": 1, "minute": 60, "hour": 3600,
            "day": 86400, "week": 604800, "month": 2592000}[unit]
    count = int(seconds // step)
    if unit == "second" and count < 10:
        return "just now"
    count = max(count, 1)
    return f"{count} {unit}{'s' if count != 1 else ''} ago"


def session_title(session: dict) -> str:
    """What the row is called. The name if it was given one, otherwise the
    instruction it started with — the id is a hex blob nobody recognises, so
    it goes in the detail line instead of the headline."""
    name = (session.get("name") or "").strip()
    if name:
        return name
    task = " ".join((session.get("task") or "").split())
    if task:
        return task if len(task) <= 72 else task[:71] + "…"
    return session.get("id", "")


def session_detail(session: dict, now: datetime = None) -> str:
    parts = [relative_age(session.get("updated_at", ""), now)]
    if session.get("model"):
        parts.append(session["model"])
    count = session.get("messages")
    if count:
        parts.append(f"{count} message{'s' if count != 1 else ''}")
    parts.append(session.get("id", ""))
    return "  ·  ".join(p for p in parts if p)


def matches(session: dict, query: str) -> bool:
    """Every whitespace-separated word has to appear somewhere in the row —
    name, task, id or model. Words rather than a substring so "auth fix"
    finds a session however those two words are arranged in its task."""
    haystack = " ".join(str(session.get(field) or "") for field in
                         ("name", "task", "id", "model", "status")).lower()
    return all(word in haystack for word in query.lower().split())


def visible_sessions(sessions: list, query: str = "", project_root: str = None,
                      branch: str = None) -> list:
    """The rows the list is showing: scoped to `project_root` (Ctrl+A clears
    it) and to `branch` (Ctrl+T clears it), then narrowed by the search box.
    None means "don't scope by this".

    A session with no branch recorded — one from before the column existed,
    or one that ran outside a checkout — survives the branch filter rather
    than vanishing from a list it has every right to be in."""
    rows = sessions
    if project_root is not None:
        wanted = os.path.abspath(project_root)
        rows = [s for s in rows if os.path.abspath(s.get("project_root") or "") == wanted]
    if branch:
        rows = [s for s in rows if (s.get("branch") or branch) == branch]
    if query.strip():
        rows = [s for s in rows if matches(s, query)]
    return rows


def scroll_window(index: int, total: int, rows: int = VISIBLE_ROWS) -> tuple:
    """(first, last) slice bounds that keep `index` on screen without the
    list jumping around: it scrolls only once the selection would otherwise
    fall off an edge."""
    if total <= rows:
        return 0, total
    first = min(max(index - rows // 2, 0), total - rows)
    return first, first + rows


class _Placeholder(Processor):
    """The dim "Search…" that stands in an empty search box. prompt_toolkit
    has no placeholder for a bare BufferControl, and an empty box with a
    magnifier beside it doesn't say that typing filters the list."""

    def __init__(self, text: str):
        self.text = text

    def apply_transformation(self, info):
        if info.document.text:
            return Transformation(info.fragments)
        return Transformation([("class:picker.placeholder", self.text)])


class SessionPicker:
    """The list itself. Built without a terminal so the selection logic can
    be exercised directly; `run()` is the only part that needs one."""

    def __init__(self, sessions: list, project_root: str = None, branch: str = None,
                  rows: int = VISIBLE_ROWS):
        self.sessions = list(sessions)
        self.project_root = project_root
        self.branch = branch or ""
        self.rows = rows
        # Scoped to where you are standing, until Ctrl+A / Ctrl+T say otherwise.
        self.all_projects = project_root is None
        self.all_branches = not self.branch
        self.index = 0
        self.query = ""
        self._app = None
        # Nothing here belongs where you are standing: an empty list with a
        # "press Ctrl+A" hint is a worse first screen than simply showing
        # what there is. Widen the branch first, since that is the narrower
        # of the two and far more often the reason the list came up empty.
        if not self.all_branches and not self.visible:
            self.all_branches = True
        if not self.all_projects and not self.visible:
            self.all_projects = True

    # ---- state ----

    @property
    def visible(self) -> list:
        return visible_sessions(
            self.sessions, self.query,
            None if self.all_projects else self.project_root,
            None if self.all_branches else self.branch,
        )

    @property
    def selected(self) -> dict:
        rows = self.visible
        return rows[self.index] if 0 <= self.index < len(rows) else None

    def search(self, query: str):
        """Retyping the filter puts the selection back on the first match —
        leaving it where it was means Enter resumes whichever session
        happens to sit at that offset in a list you have just replaced."""
        self.query = query
        self.index = 0

    def move(self, delta: int):
        total = len(self.visible)
        if total:
            self.index = max(0, min(self.index + delta, total - 1))

    def _rescope(self, change):
        """Change what the list is scoped to, keeping the highlighted session
        highlighted where it survives the change — otherwise widening the
        list silently moves the selection onto a different session."""
        keep = self.selected
        change()
        rows = self.visible
        self.index = next((i for i, s in enumerate(rows)
                            if keep is not None and s["id"] == keep["id"]), 0)

    def toggle_projects(self):
        """Ctrl+A."""
        def flip():
            self.all_projects = not self.all_projects
        self._rescope(flip)

    def toggle_branches(self):
        """Ctrl+T. A no-op with nothing to scope to — outside a checkout
        every session already belongs to no branch in particular."""
        if not self.branch:
            return

        def flip():
            self.all_branches = not self.all_branches
        self._rescope(flip)

    # ---- rendering ----

    def title_fragments(self) -> list:
        total = len(self.visible)
        position = self.index + 1 if total else 0
        return [("class:picker.title", "  Resume session"),
                ("class:picker.count", f"  ({position} of {total})")]

    def scope_fragments(self) -> list:
        """What the list is showing, in the two dimensions you can widen."""
        project = ("All projects" if self.all_projects
                    else os.path.basename(os.path.abspath(self.project_root)))
        parts = [project]
        if self.branch:
            parts.append("all branches" if self.all_branches else self.branch)
        return [("class:picker.group", "  " + "  ·  ".join(parts))]

    def row_fragments(self, now: datetime = None) -> list:
        rows = self.visible
        if not rows:
            return [("class:picker.empty", "  Nothing matches — backspace to widen the search.")]
        first, last = scroll_window(self.index, len(rows), self.rows)
        out = []
        for position in range(first, last):
            session = rows[position]
            chosen = position == self.index
            if chosen:
                marker = ("class:picker.marker", "❯ ")
            elif position == first and first > 0:
                marker = ("class:picker.scroll", "↑ ")
            elif position == last - 1 and last < len(rows):
                marker = ("class:picker.scroll", "↓ ")
            else:
                marker = ("class:picker.scroll", "  ")
            name_style = "class:picker.name.selected" if chosen else "class:picker.name"
            out += [marker, (name_style, session_title(session)), ("", "\n")]
            out += [("class:picker.detail", f"  {session_detail(session, now)}"), ("", "\n")]
            if position != last - 1:
                out.append(("", "\n"))
        return out

    def hint_fragments(self) -> list:
        """Two lines, split by hand rather than left to wrap: what to press
        is on the first, what it is scoped to on the second. Character
        wrapping is all prompt_toolkit offers, and it will happily break
        "cancel" across a line end two columns from the edge."""
        moving = ["↑↓ to move", "Enter to resume", "Type to search", "Esc to cancel"]
        scoping = ["Ctrl+A for " + ("only this project" if self.all_projects else "all projects")]
        if self.branch:
            scoping.append("Ctrl+T for "
                            + ("only this branch" if self.all_branches else "all branches"))
        return [("class:picker.hint", "  " + "  ·  ".join(moving)),
                ("", "\n"),
                ("class:picker.hint", "  " + "  ·  ".join(scoping))]

    # ---- the terminal part ----

    def _build(self) -> Application:
        from . import ui

        one = Dimension.exact(1)
        buffer = Buffer(multiline=False,
                         on_text_changed=lambda b: (self.search(b.text), self.invalidate()))
        search = Window(BufferControl(buffer=buffer, input_processors=[_Placeholder("Search…")]),
                         height=one, wrap_lines=False)

        def edge(left: str, right: str):
            return VSplit([
                Window(FormattedTextControl([("class:picker.border", left)]), width=one, height=one),
                Window(char="─", height=one, style="class:picker.border"),
                Window(FormattedTextControl([("class:picker.border", right)]), width=one, height=one),
            ], height=one)

        box = HSplit([
            edge("╭", "╮"),
            VSplit([
                Window(FormattedTextControl([("class:picker.border", "│ ")]),
                        width=Dimension.exact(2), height=one),
                Window(FormattedTextControl([("class:picker.icon", "⌕ ")]),
                        width=Dimension.exact(2), height=one),
                search,
                Window(FormattedTextControl([("class:picker.border", " │")]),
                        width=Dimension.exact(2), height=one),
            ], height=one),
            edge("╰", "╯"),
        ])

        layout = Layout(HSplit([
            Window(char="─", height=one, style="class:picker.border"),
            Window(height=one),
            Window(FormattedTextControl(self.title_fragments), height=one),
            Window(height=one),
            box,
            Window(height=one),
            Window(FormattedTextControl(self.scope_fragments), height=one),
            Window(height=one),
            Window(FormattedTextControl(self.row_fragments), always_hide_cursor=True,
                    height=Dimension(min=1, max=self.rows * 3)),
            Window(height=one),
            # Two lines, because the hints only fit on one in a wide
            # terminal and a narrow one should wrap them rather than cut
            # the last one in half.
            Window(FormattedTextControl(self.hint_fragments), wrap_lines=True,
                    always_hide_cursor=True, height=Dimension(min=1, max=3)),
        ]), focused_element=search)

        keys = KeyBindings()

        @keys.add("up")
        def _up(event):
            self.move(-1)

        @keys.add("down")
        def _down(event):
            self.move(1)

        @keys.add("pageup")
        def _page_up(event):
            self.move(-self.rows)

        @keys.add("pagedown")
        def _page_down(event):
            self.move(self.rows)

        @keys.add("c-a")
        def _scope_projects(event):
            self.toggle_projects()

        @keys.add("c-t")
        def _scope_branches(event):
            self.toggle_branches()

        @keys.add("enter", filter=Condition(lambda: self.selected is not None))
        def _accept(event):
            event.app.exit(result=self.selected["id"])

        @keys.add("escape", eager=True)
        @keys.add("c-c")
        @keys.add("c-d")
        def _cancel(event):
            event.app.exit(result=None)

        return Application(layout=layout, key_bindings=keys, style=ui._PROMPT_STYLE,
                            full_screen=False, erase_when_done=True)

    def invalidate(self):
        if self._app is not None:
            self._app.invalidate()

    async def run(self):
        """Show the list and return the chosen session's id, or None if it
        was dismissed."""
        self._app = self._build()
        try:
            return await self._app.run_async()
        finally:
            self._app = None


async def pick_session(sessions: list, project_root: str = None, branch: str = None):
    return await SessionPicker(sessions, project_root, branch).run()
