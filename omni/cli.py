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
from dataclasses import replace
from typing import List, Optional

import typer

from .agent import CodingAgent, SYSTEM_PROMPT
from .config import (
    AgentConfig, env_setting, find_setting, save_setting, saved_setting,
    SETTINGS, SETTINGS_BY_NAME,
)
from .llm_client import DEFAULT_BASE_URL, LLMError, chat, list_models
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
    "/copy": "copy this agent's transcript to the system clipboard",
    "/reasoning": "show a reply's chain of thought in full — /reasoning [n]",
    "/agent": "the agents in this session — /agent <n> switches, /agent close <n> drops one",
    "/expand ": "reprint one tool call whole, arguments and result — /expand <n>",
    "/btw ": "ask a quick side question without touching this session's history",
    "/model": "list models available on the LLM server and switch (the choice is saved; "
               "also populates /model <name> below)",
    "/config": "every saved setting at once — /config reset restores all of them to defaults",
    "/mcp": "show connected MCP servers, connect time, and tool counts",
    "/mcp restart ": "reconnect an MCP server after changing it — /mcp restart <name|all>",
    "/mcp tools ": "list the tools one MCP server exposes — /mcp tools <name>",
    "/mcp remove ": "disconnect a server and stop loading it — /mcp remove <name>",
    "/resources": "list resources published by connected MCP servers — /resources <uri> reads one",
}

# One command per saved preference, off the same registry that defines what
# the settings are and how they're parsed — so a new row there is a new
# command here, listed in the completion menu, without a second list to keep
# in step. /model is the exception already spelled out above: bare it opens
# the picker rather than reporting a value.
_STATIC_COMMANDS.update({
    f"/{setting.name}": f"{setting.summary} — /{setting.name} <value> sets and saves it, "
                        f"/{setting.name} reset restores the default"
    for setting in SETTINGS if setting.name != "model"
})

app = typer.Typer(add_completion=False, help="Coding agent (Qwen Coder or any OpenAI-compatible model)")


@app.command()
def main(
    task: Optional[str] = typer.Argument(
        None, help="What you want the agent to do. Optional with --resume (continues "
                   "with no new instruction) or --list-sessions.",
    ),
    project_root: str = typer.Option(".", "--project-root", "-p", help="Directory the agent is scoped to"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model name to drive the agent. This session only — it overrides the model saved "
             "by /model without replacing it. Omit to use the saved model; with none saved, "
             "the first model the LLM server lists is used, so there is nothing to pass on a "
             "working server.",
    ),
    llm_host: Optional[str] = typer.Option(
        None, "--llm-host", help="OpenAI-compatible server URL (defaults to $LLM_HOST or http://localhost:11434)",
    ),
    llm_api_key: Optional[str] = typer.Option(
        None, "--llm-api-key",
        help="Bearer token if the LLM server sits behind an authenticated proxy "
             "(defaults to $LLM_API_KEY — prefer the env var over this flag "
             "so the key doesn't end up in your shell history).",
    ),
    llm_timeout: Optional[float] = typer.Option(
        None, "--llm-timeout",
        help="Per-request timeout (seconds) for calls to the LLM server (chat, intent parsing, "
             "history compaction). Raise this if you're seeing repeated retries with a slow/large "
             "local model — that's usually a client-side timeout, not the server being unreachable.",
    ),
    max_steps: Optional[int] = typer.Option(
        None, "--max-steps", help="Hard cap on agent loop iterations "
                                  f"(default {AgentConfig.max_steps}, or whatever /max-steps saved)",
    ),
    subagent_model: Optional[str] = typer.Option(
        None, "--subagent-model",
        help="Model for subagents the agent spawns (defaults to --model). Exploration and "
             "review are where a smaller, faster model pays off.",
    ),
    subagent_max_steps: Optional[int] = typer.Option(
        None, "--subagent-max-steps",
        help="Step cap for one subagent, separate from --max-steps so a runaway subagent "
             "can't eat the whole run's allowance.",
    ),
    safe_tool: List[str] = typer.Option(
        [], "--safe-tool",
        help="Auto-approve one extra tool, named as the model sees it (a custom server's "
             'tools are namespaced, e.g. "docs__search"). Repeatable. The built-in read-only '
             "tools are auto-approved already; everything else prompts unless --auto-approve.",
    ),
    auto_approve: bool = typer.Option(
        False, "--auto-approve",
        help="Skip human approval for write/edit/shell tools. Only use in an "
             "already-isolated environment (container/VM). Overridden if intent parsing flags the task high-risk.",
    ),
    system_prompt: Optional[str] = typer.Option(
        None, "--system-prompt",
        help="Replace the built-in system prompt with this text. This session only — it "
             "overrides the prompt saved by /system-prompt without replacing it. Omit to use "
             "the saved prompt, or the built-in one if none is saved, which is what tells the "
             "model the tool discipline the loop expects (prefer edit_file over write_file, "
             "finish with plain text, save_memory for durable facts) — a replacement should "
             "cover the same ground.",
    ),
    system_prompt_file: Optional[str] = typer.Option(
        None, "--system-prompt-file",
        help="Same as --system-prompt, read from a file (easier for anything multi-line). "
             "Mutually exclusive with --system-prompt.",
    ),
    theme_color: Optional[str] = typer.Option(
        None, "--theme-color",
        help="Accent colour for the UI as a hex value (e.g. --theme-color '#00b4d8'). "
             "Colours the prompt, spinners, the shimmer on a running label, tool calls, "
             "the agent tree and every panel border. This session only — it overrides the "
             "colour saved by /theme-color without replacing it. Omit to use the saved "
             "colour, or the built-in accent if none is saved.",
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
    context_char_budget: Optional[int] = typer.Option(
        None, "--context-char-budget",
        help="Rough character budget (not tokens) for the running conversation. Once exceeded, "
             "history is compacted — an LLM call summarizes everything except the system+task "
             "messages and the most recent --compact-keep-last messages. This session only — it "
             "overrides the budget saved by /context-char-budget without replacing it. Omit to "
             f"use the saved budget, or {AgentConfig.context_char_budget} if none is saved.",
    ),
    compact_keep_last: Optional[int] = typer.Option(
        None, "--compact-keep-last",
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
    mcp_connect_timeout: Optional[float] = typer.Option(
        None, "--mcp-connect-timeout",
        help="Seconds one MCP server gets to complete its handshake before the session "
             "carries on without it. A server that starts but never speaks MCP (or an "
             "unreachable URL) is reported as failed instead of blocking startup; "
             "/mcp shows the error and /mcp restart <name> retries it.",
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
        help="With --add-mcp-server or --mcp-server: don't expose this server's tools to the "
             "model up front. "
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

    if system_prompt is not None and system_prompt_file is not None:
        typer.echo("Error: pass --system-prompt or --system-prompt-file, not both.", err=True)
        raise typer.Exit(code=1)

    if system_prompt_file is not None:
        try:
            system_prompt = open(system_prompt_file, encoding="utf-8").read()
        except OSError as e:
            typer.echo(f"Error: could not read --system-prompt-file {system_prompt_file!r}: {e}",
                        err=True)
            raise typer.Exit(code=1)
        if not system_prompt.strip():
            # Silently falling back to the built-in prompt would look like the
            # flag was ignored, which is worse than refusing.
            typer.echo(f"Error: --system-prompt-file {system_prompt_file!r} is empty.", err=True)
            raise typer.Exit(code=1)

    # Every preference in the settings registry is resolved the same way:
    # the flag if it was passed, else what the matching slash command saved,
    # else the built-in default. The flags are session overrides and are
    # deliberately never written back — trying a model or a colour out for one
    # run must not quietly become the setting for every run after it.
    preferences = _resolve_preferences({
        "model": model,
        "llm_host": llm_host,
        "llm_timeout_s": llm_timeout,
        "max_steps": max_steps,
        "subagent_model": subagent_model,
        "subagent_max_steps": subagent_max_steps,
        # --skip-intent-parsing is the flag; parse_intent is the preference,
        # so the flag only speaks when it is actually passed.
        "parse_intent": False if skip_intent_parsing else None,
        "intent_model": intent_model,
        "compact_model": compact_model,
        "compact_keep_last": compact_keep_last,
        "context_char_budget": context_char_budget,
        "embedding_model": embedding_model,
        "mcp_connect_timeout_s": mcp_connect_timeout,
        "system_prompt": system_prompt,
        "theme_color": theme_color,
    })

    # Nothing chose a model — not the flag, not the settings file, not
    # $DEFAULT_LLM_MODEL — so ask the server what it has. There is no name
    # compiled in to fall back on, by design: any default is the wrong model
    # on every install that doesn't happen to run it, and the server already
    # knows the right answer.
    host = preferences["llm_host"] or DEFAULT_BASE_URL
    if not preferences["model"]:
        preferences["model"] = _discover_model(preferences["llm_host"], llm_api_key or "")
        if preferences["model"]:
            typer.echo(f"No model set — using {preferences['model']}, the first one {host} "
                       "lists. /model switches and saves.")

    if not preferences["model"]:
        # Refusing here is the whole point of having no default: the
        # alternative is a run that gets as far as its first call and fails
        # inside the server, where the fix is much harder to read off.
        typer.echo(f"Error: no model set, and {host} couldn't be asked which it has. "
                   "Start the LLM server, or name a model with --model <name>, "
                   "$DEFAULT_LLM_MODEL, or /model (which saves it for every later "
                   "run).", err=True)
        raise typer.Exit(code=1)

    if preferences["theme_color"]:
        try:
            # Applied before anything renders — the full-screen app copies the
            # style map when it's built.
            from . import ui
            ui.set_accent(preferences["theme_color"])
        except ImportError:
            pass

    try:
        extra_mcp_servers = parse_mcp_server_specs(mcp_server)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    if defer:
        # --defer used to be honored only by --add-mcp-server (handled above,
        # which exits), so pairing it with --mcp-server silently did nothing.
        for spec in extra_mcp_servers.values():
            spec["defer"] = True
        if not extra_mcp_servers:
            typer.echo("Warning: --defer had no effect — it applies to a server registered with "
                       "--add-mcp-server or passed with --mcp-server.", err=True)

    # Explicit --mcp-config wins; otherwise auto-load the global registry
    # (~/.omni-coder/omni-coder-settings.json) if it exists, so servers added once via
    # --add-mcp-server are available on every run without any flags.
    effective_mcp_config_path = mcp_config or (
        default_mcp_config_path() if os.path.exists(default_mcp_config_path()) else ""
    )

    cfg = AgentConfig(
        llm_api_key=llm_api_key or "",
        project_root=project_root,
        auto_approve=auto_approve,
        log_path=log_path,
        mcp_log_path=mcp_log_path,
        db_path=db_path,
        mcp_config_path=effective_mcp_config_path,
        mcp_servers=extra_mcp_servers,
        safe_tools=AgentConfig.safe_tools + tuple(safe_tool),
        **preferences,
    )

    if task is None:
        asyncio.run(_interactive(cfg, resume, session_name))
        return

    if resume:
        _show_resumed_history(db_path, resume)

    agent = CodingAgent(cfg)
    try:
        result = asyncio.run(agent.run(task, resume_session_id=resume, session_name=session_name))
    except (ValueError, RuntimeError, LLMError) as e:
        # RuntimeError is what _call_model raises once its retries are spent
        # — i.e. "the LLM server is down", by far the most common failure
        # here. It deserves one line, not a stack trace.
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

    def session_title() -> str:
        """What the terminal window/tab is named. The session's own name where
        there is one, otherwise the id it was given — the "(resumed)" suffix
        that session_label() carries is noise in a window title."""
        return session_name or resume or session_id or "omni"

    def session_label() -> str:
        # Prefer whatever human-chosen name identifies this session — the
        # --session-name given for a new one, or the --resume value (which
        # may itself be a name) — over the opaque hex id the DB assigns,
        # so a name typed at startup never gets silently swapped for an id.
        if resume:
            return f"{resume} (resumed)"
        return session_name or session_id or "(new)"

    # The header is drawn once, at startup, and never redrawn: the two things
    # it used to be redrawn for — the model changing and the session gaining
    # its real id after turn one — are both visible without it. The session
    # name rides on the input frame's chip (which is rebuilt every prompt),
    # and a /model switch echoes the new name. Redrawing meant a second copy
    # of the box landing in the middle of the transcript.
    commands = dict(_STATIC_COMMANDS)  # mutated in place below once MCP prompts are discovered

    tui = _make_tui(commands, session_label(), cfg.model)
    _set_title(session_title())
    # In a full-screen session this is the transcript's first block; without
    # one it is printed to the terminal, as before.
    _print_header(cfg, session_label())
    if tui is None:
        _echo(f"Interactive mode (model: {cfg.model}). Type a task, /sessions to list, "
                   "/compact to summarize a long session's history, /model to list/switch models, "
                   "/exit to quit. Ctrl+C interrupts the current turn without leaving the session.")

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

    # Main is pane 0. Every pane owns its agent and its session, which is what
    # lets several run at once without their transcripts or histories mixing.
    main_pane = tui.main if tui is not None else _PlainPane(session_label())
    main_pane.agent = agent
    main_pane.session_id = session_id

    async with MCPToolClient(cfg.project_root, mcp_config_path=cfg.mcp_config_path or None,
                              extra_servers=cfg.mcp_servers or None,
                              embedding_model=cfg.embedding_model or None,
                              llm_host=cfg.llm_host or None,
                              llm_api_key=cfg.llm_api_key or None,
                              mcp_log_path=cfg.mcp_log_path,
                              connect_timeout_s=cfg.mcp_connect_timeout_s,
                              builtin_env=cfg.tool_server_env(),
                              spawn_agent=_make_spawner(cfg, tui, main_pane)) as client:
        await client.list_llm_tools()  # populate tool counts for /mcp before any task runs
        for e in client.server_status():
            if not e["connected"]:
                _echo(f"Warning: MCP server {e['name']!r} failed to connect: {e['error']}", err=True)

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
            if server != "built-in":
                commands[f"/mcp remove {server}"] = "disconnect and stop loading this server"
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

        # The spawner needs the live client and the dispatcher, both of which
        # only exist inside this block.
        _RUNTIME["client"] = client

        def _turn_finished(pane):
            """Run whatever was typed at this pane while it was working."""
            if pane.pending:
                text, attachments = pane.pending.pop(0)
                _start_turn(pane, text, attachments)

        def _start_turn(pane, text: str, attachments: list = None):
            # Images pasted at the pane belong to the line that was typed with
            # them, so they are taken now and cleared: whatever is pasted next
            # goes with the next prompt, not this one.
            if attachments is None:
                attachments, pane.attachments = pane.attachments, []
            if pane.task is not None and not pane.task.done():
                pane.pending.append((text, attachments))
                _echo(f"queued for {pane.name!r} — it's still working")
                return
            # Recorded synchronously: the exit path waits on in-flight turns,
            # and a task only known once _run_turn starts could be dropped.
            pane.task = asyncio.ensure_future(
                _run_turn(pane, text, cfg=cfg, client=client, tui=tui,
                          session_name=session_name, on_finished=_turn_finished,
                          attachments=attachments))

        _RUNTIME["start_turn"] = _start_turn
        runner = asyncio.ensure_future(tui.run()) if tui is not None else None
        try:
            while True:
                try:
                    pane, task = await _next_instruction(tui, main_pane)
                except (EOFError, KeyboardInterrupt):
                    _echo("")
                    break
                if pane is None or task is None:   # ctrl+d, or the app exited
                    break

                # Everything below acts on the pane the line was typed at:
                # its agent, its session, its slash commands. Main is pane 0.
                agent = pane.agent
                session_id = pane.session_id
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
                        _echo(f"Deleted session {target!r}.")
                        if session_id is not None and agent.store.resolve_session_id(session_id) is None:
                            session_id = None  # the session we were resuming just got deleted
                    else:
                        _echo(f"No session found with id or name {target!r}.", err=True)
                    continue
                if task == "/copy":
                    # The full-screen app draws the transcript itself, so the
                    # terminal has no copy of it to select from — see
                    # TuiApp.copy_transcript. Without the app the transcript
                    # is ordinary scrollback and already selectable.
                    if tui is None:
                        _echo("Nothing to copy: the transcript is your terminal's own "
                                   "scrollback here, so select it there.")
                        continue
                    copied, lines = tui.copy_transcript(pane)
                    if not lines:
                        _echo("Nothing to copy yet — this agent's transcript is empty.")
                    elif copied:
                        _echo(f"Copied {lines} line{'s' if lines != 1 else ''} "
                                   "to the clipboard.")
                    else:
                        _echo("Couldn't reach a clipboard tool "
                                   "(pbcopy / wl-copy / xclip / xsel).", err=True)
                    continue
                if task == "/compact":
                    if session_id is None:
                        _echo("No active session yet — run a task first.")
                    else:
                        _echo(await agent.compact_history(session_id))
                    continue
                if task == "/expand" or task.startswith("/expand "):
                    _expand_call(agent, task[len("/expand"):].strip())
                    continue
                if task == "/reasoning" or task.startswith("/reasoning "):
                    which = task[len("/reasoning"):].strip()
                    if which:
                        _expand_reasoning(agent, which)
                        continue
                    if agent.last_reasoning:
                        try:
                            from . import ui
                            ui.reasoning_full(agent.last_reasoning)
                        except ImportError:
                            _echo(agent.last_reasoning)
                    else:
                        _echo("No reasoning recorded yet — no reply this session "
                                   "carried a reasoning_content field.")
                    continue
                if task == "/agent" or task.startswith("/agent "):
                    _agent_command(tui, task[len("/agent"):].strip())
                    continue
                if task == "/mcp":
                    _print_mcp_status(client.server_status())
                    continue
                if task.startswith("/mcp tools"):
                    target = task[len("/mcp tools"):].strip()
                    if not target:
                        _echo("Usage: /mcp tools <name>  —  names: "
                                   f"{', '.join(client.server_names())}", err=True)
                        continue
                    try:
                        tools = await client.server_tools(target)
                    except ValueError as e:
                        _echo(f"Error: {e}", err=True)
                        continue
                    _print_server_tools(target, tools)
                    continue
                if task.startswith("/mcp remove"):
                    target = task[len("/mcp remove"):].strip()
                    if not target:
                        _echo("Usage: /mcp remove <name>  —  names: "
                                   f"{', '.join(client.server_names())}", err=True)
                        continue
                    try:
                        removed = await client.remove_server(target)
                    except ValueError as e:
                        _echo(f"Error: {e}", err=True)
                        continue
                    # Gone from this session; also unregister it if it came
                    # from the settings file, otherwise it returns next run.
                    path = default_mcp_config_path()
                    registered = load_mcp_config(path) if os.path.exists(path) else {}
                    if removed in registered:
                        del registered[removed]
                        save_mcp_config(path, registered)
                        _echo(f"Removed MCP server {removed!r} — disconnected now, and "
                                   f"unregistered from {path}.")
                    else:
                        _echo(f"Disconnected MCP server {removed!r} for this session. It "
                                   "wasn't in the settings file, so nothing to unregister.")
                    for stale in [c for c in commands
                                   if c.endswith(f" {removed}") and c.startswith("/mcp ")]:
                        del commands[stale]
                    await refresh_prompt_commands()
                    _print_mcp_status(client.server_status())
                    continue
                if task.startswith("/mcp restart"):
                    target = task[len("/mcp restart"):].strip()
                    if not target:
                        _echo("Usage: /mcp restart <name|all>  —  names: "
                                   f"{', '.join(client.server_names())}", err=True)
                        continue
                    targets = client.server_names() if target == "all" else [target]
                    for name in targets:
                        try:
                            entry = await _restart_mcp_server(client, name)
                        except ValueError as e:
                            _echo(f"Error: {e}", err=True)
                            continue
                        if entry["connected"]:
                            _echo(f"Restarted MCP server {entry['name']!r} "
                                       f"({entry['tool_count']} tools).")
                        else:
                            _echo(f"MCP server {entry['name']!r} failed to reconnect: "
                                       f"{entry['error']}", err=True)
                    await refresh_prompt_commands()  # a restarted server may expose different prompts
                    _print_mcp_status(client.server_status())
                    continue
                if task == "/resources" or task.startswith("/resources "):
                    target = task[len("/resources "):].strip() if task.startswith("/resources ") else ""
                    try:
                        resources = await client.list_resources(include_templates=True)
                    except Exception as e:
                        _echo(f"Error listing resources: {e}", err=True)
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
                        _echo(f"Error reading resource {target!r}: {e}", err=True)
                        continue
                    try:
                        from . import ui
                        ui.resource_content(target, content)
                    except ImportError:
                        _echo(f"--- {target} ---\n{content}")
                    continue
                if task == "/btw" or task.startswith("/btw "):
                    # A one-off question at the idle prompt: answered by its
                    # own stateless chat() call, never submitted to the agent
                    # as a task and never added to this session's history.
                    question = task[len("/btw"):].strip()
                    if question:
                        await _handle_btw(cfg, question)
                    else:
                        _echo("Usage: /btw <question>")
                    continue
                if task == "/model":
                    try:
                        models = await list_models(cfg.llm_host or None, cfg.llm_api_key or None)
                    except LLMError as e:
                        _echo(f"Error: {e}", err=True)
                        continue
                    for m in models:
                        commands[f"/model {m}"] = "switch to this model"

                    if prompt_session is None or not models:
                        # No prompt_toolkit (plain input() fallback), or the
                        # server returned no models: fall back to a static
                        # list — pick with "/model <name>" instead.
                        _echo(f"Current model: {cfg.model}")
                        for m in models:
                            _echo(f"  {'* ' if m == cfg.model else '  '}{m}")
                        continue

                    from prompt_toolkit.shortcuts import radiolist_dialog
                    selected = await radiolist_dialog(
                        title="Select model",
                        text=f"Current: {cfg.model}  (↑/↓ to move, Enter to select, Esc to cancel)",
                        values=[(m, m) for m in models],
                        default=cfg.model if cfg.model in models else None,
                    ).run_async()
                    if selected and selected != cfg.model:
                        # Through the setting, not straight onto the config:
                        # picking a model from the list is the same act as
                        # typing /model <name>, and should be remembered the
                        # same way.
                        _setting_command(cfg, tui, SETTINGS_BY_NAME["model"], selected)
                    continue
                verb, _, rest = task.partition(" ")
                if verb == "/config":
                    _config_command(cfg, tui, rest.strip())
                    continue
                # Every saved preference is dispatched off the registry, which
                # is also what forgives the underscored spellings: the names
                # are read off the AgentConfig fields they set, which are
                # underscored. They land here rather than falling through to
                # the MCP-prompt lookup below, which would call them unknown.
                setting = find_setting(verb)
                if setting is not None:
                    _setting_command(cfg, tui, setting, rest.strip())
                    continue
                if task.startswith("/"):
                    prompt_name, _, rest = task[1:].partition(" ")
                    if prompt_name in prompts:
                        arg_specs = prompts[prompt_name]["arguments"]
                        try:
                            values = shlex.split(rest)
                        except ValueError as e:
                            _echo(f"Error parsing arguments: {e}", err=True)
                            continue
                        if len(values) > len(arg_specs):
                            names = ", ".join(a["name"] for a in arg_specs) or "(none)"
                            _echo(
                                f"Error: /{prompt_name} takes at most {len(arg_specs)} "
                                f"argument(s): {names}", err=True,
                            )
                            continue
                        # MCP prompt arguments are string-typed (dict[str, str]) — shlex.split
                        # already yields plain strings, so no coercion is needed here.
                        prompt_args = {a["name"]: v for a, v in zip(arg_specs, values)}
                        missing = [a["name"] for a in arg_specs if a["required"] and a["name"] not in prompt_args]
                        if missing:
                            _echo(f"Error: /{prompt_name} missing required argument(s): "
                                       f"{', '.join(missing)}", err=True)
                            continue
                        try:
                            task = await client.get_prompt(prompt_name, prompt_args)
                        except Exception as e:
                            _echo(f"Error resolving prompt {prompt_name!r}: {e}", err=True)
                            continue
                        _echo(f"--- resolved /{prompt_name} ---\n{task}\n")

                # Dispatched, not awaited: the input loop has to stay
                # responsive so main keeps working while you read — or type
                # at — a subagent. One turn per pane at a time; anything typed
                # at a busy pane queues for its next turn.
                _start_turn(pane, task)

        finally:
            # Turns are dispatched, so some may still be in flight. Give them
            # a moment to land, then cancel: exiting shouldn't lose a turn
            # that was about to finish, or hang on one that wasn't.
            panes = tui.panes if tui is not None else [main_pane]
            in_flight = [p.task for p in panes if p.task is not None and not p.task.done()]
            if in_flight:
                _, still_running = await asyncio.wait(in_flight, timeout=0.5)
                for pending in still_running:
                    pending.cancel()
            if tui is not None:
                tui.stop()
                if runner is not None:
                    try:
                        await runner
                    except Exception:
                        pass
                # A full-screen app restores the terminal's previous contents
                # on exit, so write the transcript out on the way: the session
                # stays in scrollback, in whatever open/closed state it was
                # left in, instead of vanishing. Before unregistering, since
                # that is what the blocks render against.
                tui.dump()
                from . import ui
                ui.use_tui(None)


# The live client and turn dispatcher live inside _interactive's `async with`;
# the spawner closure needs them, so they're handed over here rather than
# threaded through every signature.
_RUNTIME: dict = {}


class _PlainPane:
    """A pane's worth of state without a full-screen app (the no-rich
    fallback). Carries the attributes _run_turn reads, so the turn code
    doesn't care which kind it got."""

    def __init__(self, name: str, kind: str = "main"):
        self.name = name or "main"
        self.kind = kind
        self.depth = 0 if kind == "main" else 1
        self.agent = None
        self.session_id = None
        self.task = None
        self.pending: list = []
        self.attachments: list = []
        self.on_interrupt = None
        self.busy = False
        self.done = False
        self.outcome = ""
        self.error = ""


async def _run_turn(pane, task: str, *, cfg, client, tui, session_name, on_finished=None,
                    attachments: list = None):
    """One turn for one pane.

    Its output goes to that pane's transcript (see tui.current_pane), and so
    do its approvals and questions — a background subagent asking to write a
    file waits in its own pane rather than seizing whatever you were reading.

    Returns the turn's final answer, which is what a subagent reports back to
    whoever spawned it."""
    from . import ui
    token = None
    if tui is not None:
        from . import tui as tui_mod
        token = tui_mod.current_pane.set(pane)
    previous_sigint = None
    result = None
    try:
        agent = pane.agent
        run_task = asyncio.ensure_future(
            agent.run(task, resume_session_id=pane.session_id, client=client,
                      session_name=session_name if pane.kind == "main" else None,
                      show_banner=False, attachments=attachments)
        )
        if pane.task is None:
            pane.task = asyncio.current_task()   # spawned directly, not dispatched
        if tui is None:
            # Nothing is holding the terminal, so Ctrl+C really is a signal.
            previous_sigint = signal.signal(signal.SIGINT, lambda *_: run_task.cancel())
        try:
            frame = ui.turn_frame(pane.name, cfg.model, on_interrupt=run_task.cancel)
        except ImportError:
            frame = nullcontext()
        async with frame:
            try:
                result = await run_task
                pane.outcome, pane.error = "done", ""
            except asyncio.CancelledError:
                # Stop the agent too: awaiting it was cancelled, which on its
                # own would leave the run going in the background.
                run_task.cancel()
                pane.outcome, pane.error = "interrupted", ""
                try:
                    ui.interrupted()
                except ImportError:
                    _echo("\n[Interrupted — back at the prompt.]")
            except (ValueError, RuntimeError, LLMError) as e:
                # A turn failing (a bad session id, or _call_model giving up
                # after its retries because the LLM server is unreachable)
                # must not end the session over what is usually a transient
                # hiccup. Report it; the session id is kept so a retry
                # continues the same conversation.
                pane.outcome, pane.error = "error", str(e)
                _echo(f"Error: {e}", err=True)
        pane.session_id = agent.session_id or pane.session_id
        if pane.kind == "main" and pane.session_id:
            _set_title(pane.session_id)
        if result is not None:
            try:
                ui.final_result(result)
            except ImportError:
                _echo("\n=== RESULT ===")
                _echo(result)
        return result
    finally:
        pane.task = None
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        if token is not None:
            from . import tui as tui_mod
            tui_mod.current_pane.reset(token)
        if on_finished is not None:
            on_finished(pane)


def _make_spawner(cfg: AgentConfig, tui, main_pane):
    """Build the spawn_agent handler the MCP client offers to the model.

    A subagent is a pane of its own with its own agent, its own child session
    and its own step budget, run to completion. Only its final answer crosses
    back as the tool result — that is the point of the arrangement: the parent
    reasons on the conclusion instead of carrying every step in its
    context."""
    async def spawn(task: str, name: str = "", system_prompt: str = "", model: str = "") -> str:
        client = _RUNTIME.get("client")
        if client is None:
            return "ERROR: no tool session available to run a subagent in."
        parent = main_pane
        if tui is not None:
            from . import tui as tui_mod
            parent = tui_mod.current_pane.get() or tui.main
        if getattr(parent, "depth", 0) >= 1:
            return ("ERROR: a subagent can't spawn subagents. Do the work yourself, or "
                     "report back and let the main agent decide what to delegate next.")
        child_cfg = replace(
            cfg,
            model=model or cfg.subagent_model or cfg.model,
            system_prompt=system_prompt or cfg.system_prompt,
            max_steps=cfg.subagent_max_steps or cfg.max_steps,
        )
        child = CodingAgent(child_cfg)
        label = " ".join((name or task).split())[:24] or "subagent"
        if tui is not None:
            pane = tui.add_pane(label, agent=child, depth=getattr(parent, "depth", 0) + 1)
        else:
            pane = _PlainPane(label, kind="subagent")
            pane.agent = child
        try:
            answer = await _run_turn(pane, task, cfg=child_cfg, client=client, tui=tui,
                                      session_name=None)
        except Exception as e:                     # noqa: BLE001 - reported, not raised
            # Whatever went wrong is the parent's business to know about, not
            # a reason to break its turn.
            if tui is not None:
                tui.set_idle(pane=pane, done=True)
            return f"The subagent {label!r} failed to run: {e}"
        if tui is not None:
            # Green in the agent list: it landed, and its pane is still there
            # to read — the "done" state before the answer goes back to main.
            tui.set_idle(pane=pane, done=True)
        if answer is None:
            # Say which, and say it as a fact the parent can act on: a model
            # told only "it didn't finish" tends to spawn the same job again.
            if getattr(pane, "outcome", "") == "interrupted":
                return (f"The subagent {label!r} was interrupted by the user before it "
                         "finished, so there is no answer. Don't spawn it again unless "
                         "asked — check with the user, or do the work yourself.")
            if getattr(pane, "outcome", "") == "error":
                return (f"The subagent {label!r} failed: {pane.error}. Its session holds "
                         "what it managed to do; consider doing this work yourself.")
            return (f"The subagent {label!r} produced no answer. Its session holds what it "
                     "managed to do.")
        return answer

    return spawn


def _agent_command(tui, argument: str):
    """/agent — list the agents, switch to one, or close a finished one."""
    if tui is None:
        _echo("Several agents at once needs the full-screen UI (rich + prompt_toolkit).",
               err=True)
        return
    panes = tui.panes
    if not argument:
        for index, pane in enumerate(panes):
            state = ("working" if pane.busy else
                      "waiting for you" if pane.needs_you else
                      "done" if pane.done else "idle")
            here = "  ← here" if index == tui.focused else ""
            _echo(f"{index}. {pane.name}  [{state}]{here}")
        _echo("/agent <n> switches · /agent close <n> drops a finished one · ctrl+↑↓ picks")
        return
    if argument.startswith("close"):
        rest = argument[len("close"):].strip()
        try:
            index = int(rest) if rest else tui.focused
        except ValueError:
            _echo("Usage: /agent close <n>", err=True)
            return
        if not 0 <= index < len(panes):
            _echo(f"No agent {index} — they run 0–{len(panes) - 1}.", err=True)
            return
        pane = panes[index]
        if pane.busy:
            _echo(f"{pane.name!r} is still working — interrupt it first (ctrl+c in its pane).",
                   err=True)
            return
        if not tui.close_pane(pane):
            _echo("The main agent can't be closed.", err=True)
        return
    try:
        index = int(argument)
    except ValueError:
        _echo("Usage: /agent [<n> | close <n>]", err=True)
        return
    if not 0 <= index < len(panes):
        _echo(f"No agent {index} — they run 0–{len(panes) - 1}.", err=True)
        return
    tui.focus(index)


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
        _echo(f"\n[/btw] Q: {question}\nA: {answer}\n")


def _set_title(text: str):
    """Name the terminal after the session, when the UI layer is available."""
    try:
        from . import ui
        ui.set_terminal_title(text)
    except ImportError:
        pass


def _expand_call(agent, which: str):
    """/expand <n> — reprint one tool call with nothing abbreviated. The
    number is the handle the transcript shows next to each call; a terminal
    cannot make text that has already scrolled clickable."""
    log = agent.call_log
    if not log:
        _echo("No tool calls yet this session.")
        return
    if not which:
        record = log[-1]
    else:
        try:
            index = int(which)
        except ValueError:
            _echo(f"Usage: /expand <n> — a call number from the transcript "
                        f"(1–{log[-1]['index']}).", err=True)
            return
        record = next((r for r in log if r["index"] == index), None)
        if record is None:
            _echo(f"No call [{index}] in this session — the transcript numbers run "
                        f"1–{log[-1]['index']}.", err=True)
            return
    try:
        from . import ui
        ui.call_detail(record)
    except ImportError:
        _echo(f"[{record['index']}] {record['name']}({record['args']})\n{record['result']}")


def _expand_reasoning(agent, which: str):
    """/reasoning <n> — an earlier reply's chain of thought, not just the last."""
    log = agent.reasoning_log
    if not log:
        _echo("No reasoning recorded yet — no reply this session carried a "
                   "reasoning_content field.")
        return
    try:
        index = int(which)
    except ValueError:
        _echo(f"Usage: /reasoning [n] — 1–{len(log)}, or bare for the most recent.",
                    err=True)
        return
    if not 1 <= index <= len(log):
        _echo(f"No reasoning [{index}] this session — they run 1–{len(log)}.", err=True)
        return
    try:
        from . import ui
        ui.reasoning_full(log[index - 1], index)
    except ImportError:
        _echo(log[index - 1])


def _echo(text: str, err: bool = False):
    """One line of feedback from the REPL or one of its commands.

    In a full-screen session there is no free-standing stdout to write to —
    the app owns the screen — so it becomes a transcript block. Outside one it
    is an ordinary echo. Every command handler below goes through this rather
    than deciding for itself."""
    try:
        from . import ui
        if ui.tui_active():
            (ui.error if err else ui.note)(text)
            return
    except ImportError:
        pass
    typer.echo(text, err=err)


def _discover_model(host: str, api_key: str, timeout_s: float = 8.0) -> str:
    """The first model the LLM server lists, or "" if it can't say.

    Which model that is comes down to the order the server reports, and the
    only alternative — guessing from the names — would be worse: the caller
    says where the name came from, and /model overrides it permanently in one
    line. The timeout is short and separate from --llm-timeout, which is
    sized for generating tokens rather than for a GET that should answer
    immediately: a slow server must not turn startup into a wait."""
    try:
        models = asyncio.run(list_models(host or None, api_key or None, timeout=timeout_s))
    except (LLMError, OSError, RuntimeError):
        return ""
    return models[0] if models else ""


def _resolve_preferences(chosen: dict) -> dict:
    """{field: value} for every setting in the registry.

    `chosen` carries what the flags said, with None for "not passed" — which
    is why the flags that back a preference all default to None rather than to
    the value they used to hardcode. A flag that was passed is validated the
    same way the slash command validates what is typed at it, so
    --max-steps 0 is refused at the door instead of producing a run that
    ends before its first step.

    Below the flag: what /<name> saved, then the setting's environment
    variable if it has one, then the built-in default."""
    resolved = {}
    for setting in SETTINGS:
        given = chosen.get(setting.field)
        if given is not None:
            value = setting.parse(given)
            if value is None:
                typer.echo(f"Error: --{setting.name} {given!r} is not usable — expected "
                           f"{setting.hint}.", err=True)
                raise typer.Exit(code=1)
        else:
            value = saved_setting(setting)
            if value is None:
                value = env_setting(setting)
            if value is None:
                value = getattr(AgentConfig, setting.field)
        resolved[setting.field] = value
    return resolved


def _setting_command(cfg, tui, setting, argument: str):
    """One saved preference: read it, set it, or put it back to the default.

    Every setting in the registry is driven through here rather than through a
    handler of its own, so they all agree on what "reset" means, on where the
    value is written, and on what happens when the settings file can't be
    written — the value still applies to the session in hand, because a
    preference that didn't stick is worth saying rather than refusing.

    Saving is the whole point: the flags already set any of these for one run,
    and the thing they can't do is remember."""
    current = getattr(cfg, setting.field)
    default = getattr(AgentConfig, setting.field)

    if not argument:
        where = _setting_origin(setting, current)
        if where:
            where = f"  [{where}]"
        body = f"\n{_preview(current)}\n" if setting.from_file and current else " "
        _echo(f"{setting.name}: {_setting_display(setting, current)}{where} — "
              f"{setting.summary}.{body}"
              f"Set one with /{setting.name} <value> ({setting.hint}), "
              f"or /{setting.name} reset.")
        return

    if argument.lower() in ("reset", "default", "clear"):
        try:
            path = save_setting(setting.key, None)
        except OSError as e:
            _echo(f"Error: could not write the settings file: {e}", err=True)
            return
        _apply_setting(cfg, tui, setting, default)
        _echo(f"{setting.name} reset to its default "
              f"({_setting_display(setting, default)}), cleared from {path}."
              f"{_setting_scope_note(setting)}")
        return

    verb, _, rest = argument.partition(" ")
    if setting.from_file and verb.lower() == "file":
        source = rest.strip()
        if not source:
            _echo(f"Error: /{setting.name} file needs a path.", err=True)
            return
        try:
            argument = open(os.path.expanduser(source), encoding="utf-8").read()
        except OSError as e:
            _echo(f"Error: could not read {source!r}: {e}", err=True)
            return
        if not argument.strip():
            # Saving it would clear the setting rather than replace it, which
            # is not what "set it from this file" asked for.
            _echo(f"Error: {source!r} is empty — use /{setting.name} reset to go back to "
                   "the default.", err=True)
            return

    value = setting.parse(argument)
    if value is None:
        _echo(f"Error: {argument!r} isn't usable for /{setting.name} — expected "
              f"{setting.hint}.", err=True)
        return

    shown = _setting_display(setting, value)
    try:
        path = save_setting(setting.key, value)
    except OSError as e:
        _apply_setting(cfg, tui, setting, value)
        _echo(f"{setting.name} set to {shown} for this session — could not save it: {e}",
              err=True)
        return
    _apply_setting(cfg, tui, setting, value)
    _echo(f"{setting.name} → {shown}, saved to {path}.{_setting_scope_note(setting)}")


def _setting_origin(setting, value) -> str:
    """Where a setting's current value came from, or "" for the built-in
    default.

    Worth saying: a value set for this session only looks identical to a
    saved one until the next run, which is when the difference bites — and a
    value that arrived from the environment is one nothing in the settings
    file explains."""
    if value == getattr(AgentConfig, setting.field):
        return ""
    if saved_setting(setting) == value:
        return "saved"
    if env_setting(setting) == value:
        return f"${setting.env}"
    return "this session only"


def _apply_setting(cfg, tui, setting, value):
    """Put a setting's new value on the live config, and anywhere else that
    holds a copy of it. Only the accent colour has such a copy: every Rich
    renderer builds its styles from ACCENT at call time, but the full-screen
    app was handed the style map when it was constructed."""
    setattr(cfg, setting.field, value)
    if setting.field == "theme_color":
        colour = value or _default_accent()
        if colour:
            _apply_theme_color(colour, tui)
    elif setting.field == "model":
        _announce_model(value, tui)


def _setting_display(setting, value) -> str:
    """One setting's value, on one line.

    The system prompt is the reason this isn't just str(): replaying a whole
    prompt into the transcript to confirm it was saved is worse than saying
    how long it is — the top of it is shown separately, by the one command
    that has room for it."""
    if not value and not isinstance(value, bool):
        empty = setting.empty
        if setting.field == "theme_color":
            accent = _default_accent()
            return f"{empty} {accent}".strip()
        if setting.field == "system_prompt":
            return f"{empty} ({len(SYSTEM_PROMPT)} chars)"
        return empty
    if isinstance(value, bool):
        return "on" if value else "off"
    if setting.from_file:
        return f"{len(value):,} chars"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _setting_scope_note(setting) -> str:
    """Why a setting that was just saved hasn't changed anything yet.

    Without this the two that aren't live look broken: the system prompt is
    stored as a session's first message when the session is created, and the
    tool-side knobs are handed to the MCP server process as environment when
    it connects."""
    if setting.scope == "session":
        return " Takes effect in the next new session."
    if setting.scope == "run":
        return " Takes effect the next time omni starts."
    return ""


def _preview(text: str, lines: int = 6, width: int = 100) -> str:
    """The first few lines of `text`, for echoing a setting back without
    replaying a whole system prompt into the transcript."""
    body = text.strip().splitlines()
    shown = [line if len(line) <= width else line[:width - 1] + "…" for line in body[:lines]]
    if len(body) > lines:
        shown.append("…")
    return "\n".join(shown)


def _default_accent() -> str:
    """The built-in accent, or "" on a bare install where there is no UI to
    have one. /config lists every setting, theme-color included, so it must
    not be the one command that needs rich installed to print a line."""
    try:
        from . import ui
        return ui.DEFAULT_ACCENT
    except ImportError:
        return ""


def _config_command(cfg, tui, argument: str):
    """/config — every saved preference at once, and the one way to clear
    them all. A settings file is easy to accumulate and hard to remember, so
    the listing marks which values came from it."""
    if argument.lower() in ("reset", "default", "clear"):
        cleared = []
        for setting in SETTINGS:
            if saved_setting(setting) is None:
                continue
            try:
                path = save_setting(setting.key, None)
            except OSError as e:
                _echo(f"Error: could not write the settings file: {e}", err=True)
                return
            _apply_setting(cfg, tui, setting, getattr(AgentConfig, setting.field))
            cleared.append(setting.name)
        if not cleared:
            _echo("Nothing saved to reset — every setting is already at its default.")
            return
        _echo(f"Reset to defaults: {', '.join(cleared)} (cleared from {path}). "
              "Registered MCP servers are untouched.")
        return
    if argument:
        setting = find_setting(argument)
        if setting is None:
            _echo(f"Error: {argument!r} isn't a setting. /config lists them.", err=True)
            return
        _setting_command(cfg, tui, setting, "")
        return

    lines = []
    for setting in SETTINGS:
        value = getattr(cfg, setting.field)
        mark = _setting_origin(setting, value)
        shown = _setting_display(setting, value)
        lines.append(f"  /{setting.name:<22} {shown}" + (f"  [{mark}]" if mark else ""))
    _echo("Settings (/<name> <value> sets and saves one, /<name> reset restores its "
          "default, /config reset restores all):\n" + "\n".join(lines))


# Named wrappers for the settings that had a command before the registry
# existed. They are what the REPL and the tests call; the behaviour is the
# generic one.

def _theme_color_command(cfg, tui, argument: str):
    return _setting_command(cfg, tui, SETTINGS_BY_NAME["theme-color"], argument)


def _system_prompt_command(cfg, argument: str, tui=None):
    return _setting_command(cfg, tui, SETTINGS_BY_NAME["system-prompt"], argument)


def _context_char_budget_command(cfg, argument: str, tui=None):
    return _setting_command(cfg, tui, SETTINGS_BY_NAME["context-char-budget"], argument)


def _apply_theme_color(colour: str, tui=None):
    """Recolour the live UI. ui.set_accent covers every Rich renderer, since
    they build their styles from ACCENT at call time; the full-screen app is
    the exception, because it was handed the style map at construction."""
    from . import ui
    ui.set_accent(colour)
    if tui is not None:
        tui.restyle()


def _announce_model(model: str, tui=None):
    """Confirm a /model switch.

    The header isn't redrawn for it — the header is scrollback by then — so
    the frame's hint line carries the live model name, which means the app
    has to be told about the change or that line goes stale."""
    if tui is not None:
        tui.model = model
    try:
        from . import ui
        ui.model_switched(model)
    except ImportError:
        _echo(f"Switched to model {model!r}.")


def _print_header(cfg: AgentConfig, session_label: str):
    """Re-print the header box — used at REPL startup and again whenever
    the model or session identity changes (a /model switch, or the session
    getting a real id after its first turn), so the box on screen never
    goes stale."""
    try:
        from . import ui
        ui.header(session_label, cfg.project_root)
    except ImportError:
        _echo(f"[model: {cfg.model}] [session: {session_label}]")


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
        _echo(f"--- Resumed history ({len(messages)} messages) ---")
        for m in messages:
            if m.get("role") == "system":
                continue
            _echo(f"{m.get('role')}: {str(m.get('content'))[:200]}")
        _echo("--- end history ---\n")


def _make_tui(commands: dict, session_label: str, model: str):
    """The full-screen UI for an interactive session, or None to fall back to
    plain line input.

    The transcript is drawn by us rather than left in the terminal's
    scrollback: that is the only way a tool call or a reasoning block can be
    clicked open in place, since a terminal delivers mouse events only to the
    region an application draws and cannot rewrite what has already scrolled.

    Returning None (no rich/prompt_toolkit installed, or a caller opting out)
    keeps the old behaviour, printing to the terminal and reading with
    input()."""
    try:
        from . import ui
        from .tui import TuiApp
    except ImportError:
        return None
    app = TuiApp(commands, session_label, model)
    ui.use_tui(app)
    return app


async def _next_instruction(tui, main_pane):
    """The next (pane, text) typed.

    From the running app's queue when there is one — it owns the screen and
    the keyboard for the whole session, and tags each line with the pane it
    was typed at. Otherwise a plain input(), which only ever has main.
    (None, None) ends the session."""
    if tui is not None:
        return await tui.next_instruction()
    try:
        return main_pane, input("> ")
    except EOFError:
        return None, None


def _print_sessions(sessions: list):
    if not sessions:
        _echo("No saved sessions.")
        return
    try:
        from . import ui
        ui.sessions_table(sessions)
    except ImportError:
        for s in sessions:
            _echo(f"{s['id']}  {s.get('name') or '-'}  [{s['status']}]  {s['updated_at']}  {s['task'][:70]}")


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
            _echo("No resources published by the connected MCP servers.")
            return
        for uri, info in resources.items():
            kind = "template" if info.get("template") else (info.get("mime_type") or "-")
            _echo(f"{uri}  [{info.get('server', '')}]  {kind}  {info.get('description', '')}")
        _echo("Read one with /resources <uri>")


def _print_server_tools(server: str, tools: list):
    try:
        from . import ui
        ui.server_tools_table(server, tools)
    except ImportError:
        if not tools:
            _echo(f"{server} exposes no tools.")
            return
        for t in tools:
            tags = " ".join(k for k in ("internal", "deferred", "revealed") if t.get(k))
            desc = t["description"].splitlines()[0] if t["description"] else ""
            _echo(f"{t['name']}  {f'[{tags}]  ' if tags else ''}{desc[:80]}")


def _print_mcp_status(entries: list):
    try:
        from . import ui
        ui.mcp_status(entries)
    except ImportError:
        from .agent import _format_elapsed
        for e in entries:
            if e["connected"]:
                _echo(f"[OK]   {e['name']}  connected {_format_elapsed(e['connected_for'])}  "
                           f"{e['tool_count']} tools  {e['target']}")
            else:
                _echo(f"[FAIL] {e['name']}  {e['error']}", err=True)


if __name__ == "__main__":
    app()
