"""CLI entry point (Typer).

Examples:
    python cli.py "Add type hints to utils.py and run the tests" \\
        --project-root ./myrepo

    python cli.py "Fix the failing test in test_math.py" \\
        --project-root ./myrepo --auto-approve

    python cli.py --session-name refactor-utils "Add type hints to utils.py"

    python cli.py --list-sessions

    python cli.py --resume refactor-utils "also add a docstring"

    python cli.py --delete-session refactor-utils

    python cli.py                      # no task -> interactive REPL, fresh session
    python cli.py --resume refactor-utils   # no task -> interactive REPL, resumed session

    python cli.py --mcp-server "weather=python -m weather_mcp_server" \\
        --mcp-server "docs=node docs-server.js --port 4000" "Look up today's forecast"

    python cli.py --add-mcp-server "weather=python -m weather_mcp_server"  # register once,
    python cli.py "what's the forecast?"                                   # available from here on, no flags needed

    python cli.py --add-mcp-server "docs=node docs-server.js" --defer  # tools loaded on demand
    python cli.py --mcp-server "docs=node docs-server.js,defer" "..."  # same, one-off via suffix

    python cli.py --list-mcp-servers
    python cli.py --remove-mcp-server weather

    python cli.py --help
"""

import asyncio
import os
import shlex
import signal
from contextlib import nullcontext
from typing import List, Optional

import typer

from .agent import CodingAgent
from .config import AgentConfig
from .llm_client import LLMError, chat, list_models
from .mcp_client import (
    MCPToolClient, default_mcp_config_path, load_mcp_config,
    parse_mcp_server_specs, save_mcp_config,
)
from .session_store import SessionStore

_STATIC_COMMANDS = {
    "/exit": "leave the REPL",
    "/quit": "leave the REPL",
    "/sessions": "list saved sessions",
    "/delete ": "delete a saved session — /delete <id-or-name>",
    "/compact": "summarize this session's history down to a briefing",
    "/btw ": "ask a quick side question without touching this session's history",
    "/model": "list models available on the LLM server (also populates /model <name> below)",
    "/mcp": "show connected MCP servers, connect time, and tool counts",
    "/mcp restart ": "reconnect an MCP server after changing it — /mcp restart <name|all>",
    "/mcp tools ": "list the tools one MCP server exposes — /mcp tools <name>",
    "/resources": "list resources published by connected MCP servers — /resources <uri> reads one",
}

app = typer.Typer(add_completion=False, help="Coding agent (Qwen Coder or any OpenAI-compatible model)")


@app.command()
def main(
    task: Optional[str] = typer.Argument(
        None, help="What you want the agent to do. Optional with --resume (continues "
                   "with no new instruction) or --list-sessions.",
    ),
    project_root: str = typer.Option(".", "--project-root", "-p", help="Directory the agent is scoped to"),
    model: str = typer.Option("qwen3.6:35b", "--model", "-m", help="Model name to drive the agent"),
    llm_host: Optional[str] = typer.Option(
        None, "--llm-host", help="OpenAI-compatible server URL (defaults to $LLM_HOST or http://localhost:11434)",
    ),
    llm_api_key: Optional[str] = typer.Option(
        None, "--llm-api-key",
        help="Bearer token if the LLM server sits behind an authenticated proxy "
             "(defaults to $LLM_API_KEY — prefer the env var over this flag "
             "so the key doesn't end up in your shell history).",
    ),
    llm_timeout: float = typer.Option(
        AgentConfig.llm_timeout_s, "--llm-timeout",
        help="Per-request timeout (seconds) for calls to the LLM server (chat, intent parsing, "
             "history compaction). Raise this if you're seeing repeated retries with a slow/large "
             "local model — that's usually a client-side timeout, not the server being unreachable.",
    ),
    max_steps: int = typer.Option(100, "--max-steps", help="Hard cap on agent loop iterations"),
    auto_approve: bool = typer.Option(
        False, "--auto-approve",
        help="Skip human approval for write/edit/shell tools. Only use in an "
             "already-isolated environment (container/VM). Overridden if intent parsing flags the task high-risk.",
    ),
    log_path: str = typer.Option("agent_run.log", "--log-path", help="Where to write the structured run log"),
    mcp_log_path: str = typer.Option(
        AgentConfig.mcp_log_path, "--mcp-log-path",
        help="Where stderr from every stdio-transport MCP server (built-in + custom) is redirected, "
             "instead of interleaving raw subprocess output with the terminal UI.",
    ),
    skip_intent_parsing: bool = typer.Option(
        False, "--skip-intent-parsing",
        help="Skip the upfront structured-intent parse and go straight into the agent loop.",
    ),
    intent_model: Optional[str] = typer.Option(
        None, "--intent-model", help="Smaller/faster model to use just for intent parsing (defaults to --model)",
    ),
    context_char_budget: int = typer.Option(
        AgentConfig.context_char_budget, "--context-char-budget",
        help="Rough character budget (not tokens) for the running conversation. Once exceeded, "
             "history is compacted — an LLM call summarizes everything except the system+task "
             "messages and the most recent --compact-keep-last messages.",
    ),
    compact_keep_last: int = typer.Option(
        AgentConfig.compact_keep_last, "--compact-keep-last",
        help="How many of the most recent messages to keep verbatim (not summarized) when "
             "history is compacted, either automatically (--context-char-budget) or via /compact.",
    ),
    compact_model: Optional[str] = typer.Option(
        None, "--compact-model",
        help="Smaller/faster model to use just for history-compaction summaries (defaults to --model)",
    ),
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model",
        help="Embedding backend for search_tools semantic ranking against deferred MCP tool "
             'descriptions. Defaults to "nomic-local" — on-device via `pip install "nomic[local]"`, '
             "no server needed. Pass a remote OpenAI-compatible embedding model name (e.g. mxbai-embed-large) "
             'to use that instead, or "" to disable and fall back to plain keyword matching.',
    ),
    db_path: str = typer.Option(
        "agent_sessions.db", "--db-path", help="SQLite file storing session/message history",
    ),
    resume: Optional[str] = typer.Option(
        None, "--resume", help="Resume a previous session by id or --session-name instead of starting a new one",
    ),
    session_name: Optional[str] = typer.Option(
        None, "--session-name", help="Give a new session a memorable name, so you can --resume it by name later",
    ),
    list_sessions: bool = typer.Option(
        False, "--list-sessions", help="List saved sessions (id, name, status, task) and exit",
    ),
    delete_session: Optional[str] = typer.Option(
        None, "--delete-session", help="Delete a saved session (by id or --session-name) and exit",
    ),
    mcp_config: Optional[str] = typer.Option(
        None, "--mcp-config",
        help='Path to a Claude-Desktop-style MCP config file ({"mcpServers": {"name": '
             '{"command": ..., "args": [...], "env": {...}}}}) to load extra tools from, '
             "alongside the built-in ones. Their tools appear to the model as <name>__<tool>.",
    ),
    mcp_server: List[str] = typer.Option(
        [], "--mcp-server",
        help='Add one custom MCP server inline, format "name=command arg1 arg2 ...". '
             "Repeatable for multiple servers. Merged with --mcp-config if both are given "
             "(this flag wins on a name clash). Its tools appear to the model as <name>__<tool>. "
             'Append ",defer" (e.g. "name=command args...,defer") to keep this server\'s tools '
             "out of the model's default tool list — it discovers them on demand via search_tools. "
             'For a remote (http/https) server, append ",bearer=<token>" to send an '
             "Authorization: Bearer header; the value may be an env reference like "
             '"$DOCS_TOKEN", resolved at connect time so the secret stays out of shell history.',
    ),
    add_mcp_server: Optional[str] = typer.Option(
        None, "--add-mcp-server",
        help='Register a custom MCP server permanently (format "name=command arg1 arg2 ..."), '
             "then exit. Saved to the mcpServers key of ~/.omni-coder/omni-coder-settings.json and "
             "auto-loaded on every future run — no need to pass --mcp-server/--mcp-config again. "
             'Supports the same ",defer" and ",bearer=<token>" suffixes as --mcp-server (prefer '
             '",bearer=$ENV_VAR" so the token itself is never written to the settings file).',
    ),
    defer: bool = typer.Option(
        False, "--defer",
        help="With --add-mcp-server: don't expose this server's tools to the model up front. "
             "Instead a search_tools tool is offered; the model calls it with a query to load "
             "matching tools on demand, keeping unused tool schemas out of context. Default: false "
             '(equivalent to appending ",defer" to the --add-mcp-server / --mcp-server spec).',
    ),
    remove_mcp_server: Optional[str] = typer.Option(
        None, "--remove-mcp-server", help="Remove a permanently-registered MCP server by name, then exit",
    ),
    list_mcp_servers: bool = typer.Option(
        False, "--list-mcp-servers", help="List permanently-registered MCP servers and exit",
    ),
):
    """Run the coding agent on TASK inside PROJECT_ROOT. Omit TASK to enter
    an interactive session (fresh, or resumed with --resume)."""
    if add_mcp_server:
        try:
            spec = parse_mcp_server_specs([add_mcp_server])
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        (name,) = spec.keys()
        if defer:
            spec[name]["defer"] = True
        path = default_mcp_config_path()
        servers = load_mcp_config(path) if os.path.exists(path) else {}
        servers.update(spec)
        save_mcp_config(path, servers)
        suffix = " (deferred tool loading)" if spec[name].get("defer") else ""
        typer.echo(f"Registered MCP server {name!r} in {path}{suffix} — available on every run from now on.")
        raise typer.Exit()

    if remove_mcp_server:
        path = default_mcp_config_path()
        servers = load_mcp_config(path) if os.path.exists(path) else {}
        if remove_mcp_server not in servers:
            typer.echo(f"Error: no registered MCP server named {remove_mcp_server!r}.", err=True)
            raise typer.Exit(code=1)
        del servers[remove_mcp_server]
        save_mcp_config(path, servers)
        typer.echo(f"Removed MCP server {remove_mcp_server!r}.")
        raise typer.Exit()

    if list_mcp_servers:
        path = default_mcp_config_path()
        servers = load_mcp_config(path) if os.path.exists(path) else {}
        if not servers:
            typer.echo("No registered MCP servers.")
        else:
            for name, spec in servers.items():
                target = spec["url"] if "url" in spec else f"{spec['command']} {' '.join(spec.get('args', []))}"
                suffix = " [defer]" if spec.get("defer") else ""
                typer.echo(f"{name}: {target}{suffix}")
        raise typer.Exit()

    if delete_session:
        if SessionStore(db_path).delete_session(delete_session):
            typer.echo(f"Deleted session {delete_session!r}.")
        else:
            typer.echo(f"Error: no session found with id or name {delete_session!r}.", err=True)
            raise typer.Exit(code=1)
        raise typer.Exit()

    if list_sessions:
        _print_sessions(SessionStore(db_path).list_sessions())
        raise typer.Exit()

    try:
        extra_mcp_servers = parse_mcp_server_specs(mcp_server)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    # Explicit --mcp-config wins; otherwise auto-load the global registry
    # (~/.omni-coder/omni-coder-settings.json) if it exists, so servers added once via
    # --add-mcp-server are available on every run without any flags.
    effective_mcp_config_path = mcp_config or (
        default_mcp_config_path() if os.path.exists(default_mcp_config_path()) else ""
    )

    cfg = AgentConfig(
        model=model,
        llm_host=llm_host or "",
        llm_api_key=llm_api_key or "",
        llm_timeout_s=llm_timeout,
        project_root=project_root,
        max_steps=max_steps,
        auto_approve=auto_approve,
        log_path=log_path,
        mcp_log_path=mcp_log_path,
        parse_intent=not skip_intent_parsing,
        intent_model=intent_model or "",
        context_char_budget=context_char_budget,
        compact_keep_last=compact_keep_last,
        compact_model=compact_model or "",
        db_path=db_path,
        mcp_config_path=effective_mcp_config_path,
        mcp_servers=extra_mcp_servers,
        # None (flag omitted) -> AgentConfig's own default ("nomic-embed-text");
        # "" (--embedding-model "" explicitly) -> disabled.
        embedding_model=embedding_model if embedding_model is not None else AgentConfig.embedding_model,
    )

    if task is None:
        asyncio.run(_interactive(cfg, resume, session_name))
        return

    if resume:
        _show_resumed_history(db_path, resume)

    agent = CodingAgent(cfg)
    try:
        result = asyncio.run(agent.run(task, resume_session_id=resume, session_name=session_name))
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    try:
        from . import ui
        ui.final_result(result)
    except ImportError:
        typer.echo("\n=== FINAL RESULT ===")
        typer.echo(result)


async def _interactive(cfg: AgentConfig, resume: Optional[str], session_name: Optional[str]):
    """REPL: keep one MCP client open across turns (avoids re-spawning the
    tool-server subprocess every turn) and keep resuming the same session
    (fresh on turn 1, then whatever session that turn created/resumed).

    Input is read through a prompt_toolkit PromptSession wrapped in
    patch_stdout(), so the input line stays pinned to the bottom of the
    terminal — parsing/thinking spinners, panels, and results all scroll in
    the region above it instead of interleaving with the prompt. Falls back
    to a plain input() loop if rich/prompt_toolkit aren't installed."""
    agent = CodingAgent(cfg)
    session_id = resume

    def session_label() -> str:
        # Prefer whatever human-chosen name identifies this session — the
        # --session-name given for a new one, or the --resume value (which
        # may itself be a name) — over the opaque hex id the DB assigns,
        # so a name typed at startup never gets silently swapped for an id.
        if resume:
            return f"{resume} (resumed)"
        return session_name or session_id or "(new)"

    shown_header = {}  # what the on-screen header last displayed — see refresh_header

    def refresh_header():
        """Redraw the header box only when the model or session label it
        shows has actually changed. The session gains its real DB id after
        its first turn, which would otherwise redraw the box directly above
        that turn's result every time — visible as a duplicate header above
        the final "done" panel, and identical to the one already on screen
        whenever --session-name pins the label."""
        state = (cfg.model, session_label())
        if state == shown_header.get("state"):
            return
        shown_header["state"] = state
        _print_header(cfg, state[1])

    commands = dict(_STATIC_COMMANDS)  # mutated in place below once MCP prompts are discovered

    try:
        from . import ui
        from prompt_toolkit import PromptSession
        from prompt_toolkit.patch_stdout import patch_stdout
        refresh_header()
        prompt_session = PromptSession(completer=ui.SlashCommandCompleter(commands), complete_while_typing=True)
        # raw=True: pass Rich's ANSI-coded output straight through instead of
        # patch_stdout()'s default write() path, which sanitizes/escapes text
        # (it assumes plain text) and mangles embedded escape codes into
        # literal garbage like "?[32m" on the screen.
        stdout_cm = patch_stdout(raw=True)
    except ImportError:
        typer.echo(f"Interactive mode (model: {cfg.model}). Type a task, /sessions to list, "
                   "/compact to summarize a long session's history, /model to list/switch models, "
                   "/exit to quit. Ctrl+C interrupts the current turn without leaving the session.\n")
        prompt_session = None
        stdout_cm = nullcontext()

    if resume:
        _show_resumed_history(cfg.db_path, resume)
        # Resolve --resume (which may be a --session-name, not a raw id) to
        # the real DB id now rather than waiting for the first turn to set
        # it — /compact and /delete use session_id directly, and a name
        # doesn't match the DB's id column, so e.g. /compact would silently
        # find "no messages" and report nothing to compact. Left unresolved
        # (falls through to the raw value) if the name/id doesn't exist, so
        # the existing "no session found" error still surfaces from
        # agent.run() on the first turn.
        resolved = agent.store.resolve_session_id(resume)
        if resolved is not None:
            session_id = resolved

    async with MCPToolClient(cfg.project_root, mcp_config_path=cfg.mcp_config_path or None,
                              extra_servers=cfg.mcp_servers or None,
                              embedding_model=cfg.embedding_model or None,
                              llm_host=cfg.llm_host or None,
                              llm_api_key=cfg.llm_api_key or None,
                              mcp_log_path=cfg.mcp_log_path) as client:
        await client.list_llm_tools()  # populate tool counts for /mcp before any task runs
        for e in client.server_status():
            if not e["connected"]:
                typer.echo(f"Warning: MCP server {e['name']!r} failed to connect: {e['error']}", err=True)

        prompts = {}
        prompt_command_keys = set()  # which `commands` entries came from MCP prompts

        async def refresh_prompt_commands():
            """(Re)build the /-menu entries for MCP prompts. Called at startup
            and again after a restart, since a restarted server may expose a
            different set. Tracks its own keys so it never disturbs the other
            dynamic completions (/model <name>, /resources <uri>)."""
            nonlocal prompts
            prompts = await client.list_prompts()
            for stale in prompt_command_keys:
                commands.pop(stale, None)
            prompt_command_keys.clear()
            for prompt_name, info in prompts.items():
                arg_hint = " ".join(
                    f"<{a['name']}>" if a["required"] else f"[{a['name']}]" for a in info["arguments"]
                )
                key = f"/{prompt_name} "
                commands[key] = f"{info['description']} {arg_hint}".strip()
                prompt_command_keys.add(key)

        await refresh_prompt_commands()
        for server in client.server_names():
            commands[f"/mcp restart {server}"] = "reconnect this MCP server"
            commands[f"/mcp tools {server}"] = "list the tools this MCP server exposes"
        commands["/mcp restart all"] = "reconnect every MCP server"

        try:
            # Best-effort: some LLM servers don't expose /v1/models. Register
            # each model name as its own "/model <name>" completion so typing
            # "/model " pops a pickable list — /model (bare) below refreshes
            # this same set, in case models changed since startup.
            for m in await list_models(cfg.llm_host or None, cfg.llm_api_key or None):
                commands[f"/model {m}"] = "switch to this model"
        except LLMError:
            pass

        with stdout_cm:
            while True:
                try:
                    task = await _read_task(prompt_session)
                except (EOFError, KeyboardInterrupt):
                    typer.echo()
                    break

                task = task.strip()
                if not task:
                    continue
                if task in ("/exit", "/quit"):
                    break
                if task == "/sessions":
                    _print_sessions(agent.store.list_sessions())
                    continue
                if task.startswith("/delete "):
                    target = task[len("/delete "):].strip()
                    if agent.store.delete_session(target):
                        typer.echo(f"Deleted session {target!r}.")
                        if session_id is not None and agent.store.resolve_session_id(session_id) is None:
                            session_id = None  # the session we were resuming just got deleted
                    else:
                        typer.echo(f"No session found with id or name {target!r}.", err=True)
                    continue
                if task == "/compact":
                    if session_id is None:
                        typer.echo("No active session yet — run a task first.")
                    else:
                        typer.echo(await agent.compact_history(session_id))
                    continue
                if task == "/mcp":
                    _print_mcp_status(client.server_status())
                    continue
                if task.startswith("/mcp tools"):
                    target = task[len("/mcp tools"):].strip()
                    if not target:
                        typer.echo("Usage: /mcp tools <name>  —  names: "
                                   f"{', '.join(client.server_names())}", err=True)
                        continue
                    try:
                        tools = await client.server_tools(target)
                    except ValueError as e:
                        typer.echo(f"Error: {e}", err=True)
                        continue
                    _print_server_tools(target, tools)
                    continue
                if task.startswith("/mcp restart"):
                    target = task[len("/mcp restart"):].strip()
                    if not target:
                        typer.echo("Usage: /mcp restart <name|all>  —  names: "
                                   f"{', '.join(client.server_names())}", err=True)
                        continue
                    targets = client.server_names() if target == "all" else [target]
                    for name in targets:
                        try:
                            entry = await _restart_mcp_server(client, name)
                        except ValueError as e:
                            typer.echo(f"Error: {e}", err=True)
                            continue
                        if entry["connected"]:
                            typer.echo(f"Restarted MCP server {entry['name']!r} "
                                       f"({entry['tool_count']} tools).")
                        else:
                            typer.echo(f"MCP server {entry['name']!r} failed to reconnect: "
                                       f"{entry['error']}", err=True)
                    await refresh_prompt_commands()  # a restarted server may expose different prompts
                    _print_mcp_status(client.server_status())
                    continue
                if task == "/resources" or task.startswith("/resources "):
                    target = task[len("/resources "):].strip() if task.startswith("/resources ") else ""
                    try:
                        resources = await client.list_resources(include_templates=True)
                    except Exception as e:
                        typer.echo(f"Error listing resources: {e}", err=True)
                        continue
                    # Keep "/resources <uri>" completions in sync with what's
                    # actually published right now, not just at startup.
                    for stale in [c for c in commands if c.startswith("/resources ")]:
                        del commands[stale]
                    for uri, info in resources.items():
                        if not info.get("template"):
                            commands[f"/resources {uri}"] = info.get("description") or "read this resource"

                    if not target:
                        _print_resources(resources)
                        continue
                    try:
                        content = await client.read_resource(target)
                    except Exception as e:
                        typer.echo(f"Error reading resource {target!r}: {e}", err=True)
                        continue
                    try:
                        from . import ui
                        ui.resource_content(target, content)
                    except ImportError:
                        typer.echo(f"--- {target} ---\n{content}")
                    continue
                if task == "/btw" or task.startswith("/btw "):
                    # Also handled inline while a task is running (the
                    # side-reader loop below) — this covers /btw typed at
                    # the idle top-level prompt, which used to fall through
                    # and get submitted as a literal task to the agent.
                    question = task[len("/btw"):].strip()
                    if question:
                        await _handle_btw(cfg, question)
                    else:
                        typer.echo("Usage: /btw <question>")
                    continue
                if task == "/model":
                    try:
                        models = await list_models(cfg.llm_host or None, cfg.llm_api_key or None)
                    except LLMError as e:
                        typer.echo(f"Error: {e}", err=True)
                        continue
                    for m in models:
                        commands[f"/model {m}"] = "switch to this model"

                    if prompt_session is None or not models:
                        # No prompt_toolkit (plain input() fallback), or the
                        # server returned no models: fall back to a static
                        # list — pick with "/model <name>" instead.
                        typer.echo(f"Current model: {cfg.model}")
                        for m in models:
                            typer.echo(f"  {'* ' if m == cfg.model else '  '}{m}")
                        continue

                    from prompt_toolkit.shortcuts import radiolist_dialog
                    selected = await radiolist_dialog(
                        title="Select model",
                        text=f"Current: {cfg.model}  (↑/↓ to move, Enter to select, Esc to cancel)",
                        values=[(m, m) for m in models],
                        default=cfg.model if cfg.model in models else None,
                    ).run_async()
                    if selected and selected != cfg.model:
                        cfg.model = selected
                        typer.echo(f"Switched to model {cfg.model!r}.")
                        refresh_header()
                    continue
                if task.startswith("/model "):
                    cfg.model = task[len("/model "):].strip()
                    typer.echo(f"Switched to model {cfg.model!r}.")
                    refresh_header()
                    continue
                if task.startswith("/"):
                    prompt_name, _, rest = task[1:].partition(" ")
                    if prompt_name in prompts:
                        arg_specs = prompts[prompt_name]["arguments"]
                        try:
                            values = shlex.split(rest)
                        except ValueError as e:
                            typer.echo(f"Error parsing arguments: {e}", err=True)
                            continue
                        if len(values) > len(arg_specs):
                            names = ", ".join(a["name"] for a in arg_specs) or "(none)"
                            typer.echo(
                                f"Error: /{prompt_name} takes at most {len(arg_specs)} "
                                f"argument(s): {names}", err=True,
                            )
                            continue
                        # MCP prompt arguments are string-typed (dict[str, str]) — shlex.split
                        # already yields plain strings, so no coercion is needed here.
                        prompt_args = {a["name"]: v for a, v in zip(arg_specs, values)}
                        missing = [a["name"] for a in arg_specs if a["required"] and a["name"] not in prompt_args]
                        if missing:
                            typer.echo(f"Error: /{prompt_name} missing required argument(s): "
                                       f"{', '.join(missing)}", err=True)
                            continue
                        try:
                            task = await client.get_prompt(prompt_name, prompt_args)
                        except Exception as e:
                            typer.echo(f"Error resolving prompt {prompt_name!r}: {e}", err=True)
                            continue
                        typer.echo(f"--- resolved /{prompt_name} ---\n{task}\n")

                # Run the turn as a Task so Ctrl+C can cancel just this turn
                # (via the SIGINT handler below) instead of killing the whole
                # REPL — a raw KeyboardInterrupt raised inside asyncio's own
                # blocking wait can otherwise escape uncaught past this loop
                # entirely. task.cancel() injects CancelledError at the
                # coroutine's next await point (model call, tool call, etc.),
                # unwinding just that turn; the MCP client and session history
                # already written to disk are untouched, so the REPL keeps going.
                run_task = asyncio.ensure_future(
                    agent.run(task, resume_session_id=session_id, client=client,
                              session_name=session_name, show_banner=False)
                )
                previous_sigint = signal.signal(signal.SIGINT, lambda *_: run_task.cancel())
                try:
                    result = await run_task
                except asyncio.CancelledError:
                    session_id = agent.session_id or session_id
                    try:
                        from . import ui
                        ui.interrupted()
                    except ImportError:
                        typer.echo("\n[Interrupted — back to prompt. You can keep chatting in this session.]")
                    continue
                except ValueError as e:
                    typer.echo(f"Error: {e}", err=True)
                    continue
                finally:
                    signal.signal(signal.SIGINT, previous_sigint)

                if agent.session_id != session_id:
                    session_id = agent.session_id
                    refresh_header()

                try:
                    from . import ui
                    ui.final_result(result)
                except ImportError:
                    typer.echo("\n=== RESULT ===")
                    typer.echo(result)


async def _handle_btw(cfg: AgentConfig, question: str):
    """/btw <question>: answer a quick side question at the idle prompt
    without touching this session's messages or history — a one-off,
    stateless chat() call, not persisted anywhere."""
    try:
        reply = await chat(
            model=cfg.model,
            messages=[
                {"role": "system", "content": "Answer concisely and directly — this is a quick "
                                               "aside the user is asking, unrelated to any coding task."},
                {"role": "user", "content": question},
            ],
            base_url=cfg.llm_host, api_key=cfg.llm_api_key, timeout=cfg.llm_timeout_s,
        )
        answer = reply.get("content") or "(empty response)"
    except LLMError as e:
        answer = f"Error: {e}"

    try:
        from . import ui
        ui.btw_answer(question, answer)
    except ImportError:
        typer.echo(f"\n[/btw] Q: {question}\nA: {answer}\n")


def _print_header(cfg: AgentConfig, session_label: str):
    """Re-print the header box — used at REPL startup and again whenever
    the model or session identity changes (a /model switch, or the session
    getting a real id after its first turn), so the box on screen never
    goes stale."""
    try:
        from . import ui
        ui.header(cfg.model, session_label)
    except ImportError:
        typer.echo(f"[model: {cfg.model}] [session: {session_label}]")


def _show_resumed_history(db_path: str, resume: str):
    """Print the conversation being resumed so it's visible on screen that
    context actually carried over — agent.run() feeds it to the model
    either way, but nothing else displays it."""
    store = SessionStore(db_path)
    session_id = store.resolve_session_id(resume)
    if session_id is None:
        return  # let agent.run() raise the proper "no session found" error
    messages = store.load_messages(session_id)
    try:
        from . import ui
        ui.history_panel(messages)
    except ImportError:
        typer.echo(f"--- Resumed history ({len(messages)} messages) ---")
        for m in messages:
            if m.get("role") == "system":
                continue
            typer.echo(f"{m.get('role')}: {str(m.get('content'))[:200]}")
        typer.echo("--- end history ---\n")


async def _read_task(prompt_session) -> str:
    if prompt_session is not None:
        from . import ui
        return await ui.prompt_task_async(prompt_session)
    return input("> ")


def _print_sessions(sessions: list):
    if not sessions:
        typer.echo("No saved sessions.")
        return
    try:
        from . import ui
        ui.sessions_table(sessions)
    except ImportError:
        for s in sessions:
            typer.echo(f"{s['id']}  {s.get('name') or '-'}  [{s['status']}]  {s['updated_at']}  {s['task'][:70]}")


async def _restart_mcp_server(client: MCPToolClient, name: str) -> dict:
    """Restart one server, with a spinner — reconnecting spawns a subprocess
    (or reopens a remote connection) and re-lists its tools, so it isn't
    instant. Returns that server's status entry."""
    try:
        from . import ui
        spinner = ui.thinking(f"Restarting {name}…")
    except ImportError:
        spinner = nullcontext()
    with spinner:
        return await client.restart_server(name)


def _print_resources(resources: dict):
    try:
        from . import ui
        ui.resources_table(resources)
    except ImportError:
        if not resources:
            typer.echo("No resources published by the connected MCP servers.")
            return
        for uri, info in resources.items():
            kind = "template" if info.get("template") else (info.get("mime_type") or "-")
            typer.echo(f"{uri}  [{info.get('server', '')}]  {kind}  {info.get('description', '')}")
        typer.echo("Read one with /resources <uri>")


def _print_server_tools(server: str, tools: list):
    try:
        from . import ui
        ui.server_tools_table(server, tools)
    except ImportError:
        if not tools:
            typer.echo(f"{server} exposes no tools.")
            return
        for t in tools:
            tags = " ".join(k for k in ("internal", "deferred", "revealed") if t.get(k))
            desc = t["description"].splitlines()[0] if t["description"] else ""
            typer.echo(f"{t['name']}  {f'[{tags}]  ' if tags else ''}{desc[:80]}")


def _print_mcp_status(entries: list):
    try:
        from . import ui
        ui.mcp_status(entries)
    except ImportError:
        from .agent import _format_elapsed
        for e in entries:
            if e["connected"]:
                typer.echo(f"[OK]   {e['name']}  connected {_format_elapsed(e['connected_for'])}  "
                           f"{e['tool_count']} tools  {e['target']}")
            else:
                typer.echo(f"[FAIL] {e['name']}  {e['error']}", err=True)


if __name__ == "__main__":
    app()
