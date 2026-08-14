"""Rich-powered terminal presentation for the coding agent.

Nothing here affects agent logic — it's purely how things are shown. If rich
isn't installed, agent.py falls back to plain print() (see its import guard).
"""

import asyncio
import json
import re
import time
import zlib

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
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

_STATUS_COLOR = {"done": "green", "running": "yellow", "max_steps": "yellow", "error": "red"}


def banner(task: str, model: str):
    console.print(Rule(f"[bold {ACCENT}]Coding Agent[/bold {ACCENT}]", style=ACCENT))
    console.print(f"[dim]model:[/dim] {model}")
    console.print(f"[dim]task:[/dim]  {task}\n")


_MASCOT_ART = """
  ▄▄▄▄▄▄▄
 █  ●  ● █
 █    ▽   █
  ▀▀▄▄▄▀▀
  ╱ ╲╱ ╲╱
""".strip("\n")


def header(model: str, session_label: str):
    """Box-framed header for interactive mode — model and session name are
    re-printed via this same function (not just shown once at startup)
    whenever either changes: a /model switch, or the session getting a real
    id after the first turn."""
    mascot = Text(_MASCOT_ART, style=ACCENT)
    info = Text()
    info.append("model:   ", style="bold")
    info.append(f"{model}\n")
    info.append("session: ", style="bold")
    info.append(f"{session_label}\n\n")
    info.append("Type a task, or / for available commands\n", style="dim")
    info.append("Ctrl+C interrupts the current turn", style="dim")

    grid = Table.grid(padding=(0, 3))
    grid.add_column()
    grid.add_column()
    grid.add_row(mascot, info)

    console.print(Panel(grid, title=f"[bold {ACCENT}]Omni Coder[/bold {ACCENT}]", border_style=ACCENT, expand=True))


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


async def prompt_task_async(session) -> str:
    """Read one line of input via a prompt_toolkit PromptSession, so the
    input line stays pinned to the bottom of the terminal — all other
    output (parsing/thinking spinners, panels, results) scrolls in the
    region above it instead of interleaving with the prompt. `session` is
    a prompt_toolkit.PromptSession the caller creates once and reuses
    across turns (so up-arrow history works). Must be called inside the
    caller's `patch_stdout()` context so Rich's output (which resolves
    sys.stdout lazily on every print) is redrawn above the pinned prompt
    instead of corrupting it."""
    console.print(Rule(style="dim"))
    return await session.prompt_async(HTML(f'<style fg="{ACCENT}"><b>❯</b></style> '))


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
                elapsed = _format_elapsed(time.monotonic() - self._start)
                self._spinner.update(text=f"{self._label} ({elapsed})")
                self._live.refresh()
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass  # cosmetic ticker — never let it surface an error over the real work

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
    """Spinner shown while waiting on a model call or tool execution. Safe
    to use around an `await` — rich's status display refreshes on its own
    thread, so it doesn't block the event loop. Ticks a live elapsed-time
    counter for as long as it's open (see _TickingSpinner)."""
    return _TickingSpinner(f"[bold {ACCENT}]{label}[/bold {ACCENT}]")


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
    the real tool calls do, just via the read-only _preview_* tools."""
    emoji = _emoji_for(name)
    if name == "edit_file":
        path = args.get("path", "")
        ok, preview = await client.preview_edit(path, args.get("old_str", ""), args.get("new_str", ""))
        if not ok:
            console.print(Panel(f"[red]{preview}[/red]", title=f"{emoji} edit_file: {path}", border_style="red"))
            return False
        added, removed = _diff_stats(preview)
        title = f"{emoji} edit_file: {path}  [green]+{added}[/green] [red]-{removed}[/red]"
        console.print(Panel(_render_diff(preview), title=title, border_style="yellow"))
    elif name == "write_file":
        path = args.get("path", "")
        is_new, preview = await client.preview_write(path, args.get("content", ""), args.get("overwrite", False))
        added, removed = _diff_stats(preview)
        label = "new" if is_new else "overwrite"
        title = f"{emoji} write_file ({label}): {path}  [green]+{added}[/green] [red]-{removed}[/red]"
        console.print(Panel(_render_diff(preview), title=title, border_style="green" if is_new else "yellow"))
    elif name == "run_shell":
        cmd = args.get("command", "")
        console.print(Panel(Syntax(cmd, "bash", theme="ansi_dark"),
                             title=f"{emoji} run_shell", border_style=ACCENT))
    else:
        console.print(Panel(json.dumps(args, indent=2), title=f"{emoji} {name}", border_style="white"))

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
    for c in calls:
        console.print(f"[{ACCENT}]{_call_str(c['name'], c['args'])}[/{ACCENT}]")
        line, diff_body = _result_line(c["name"], c["args"], str(c["result"]), c["ok"], c.get("duration"))
        console.print(f"  [dim]⎿[/dim]  {line}")
        if diff_body is not None:
            console.print(Padding(_render_diff(diff_body), (0, 0, 0, 5)))


def final_result(text: str):
    console.print(Rule(f"[bold {ACCENT}]Done[/bold {ACCENT}]", style=ACCENT))
    console.print(Panel(Markdown(text), border_style=ACCENT))


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
                console.print(Panel(Markdown(content), border_style=ACCENT, expand=False))
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
