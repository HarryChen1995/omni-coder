"""Agent loop: model <-> MCP tool server, with the guardrails a first draft skips.

Tools now live in mcp_server.py and are reached through mcp_client.MCPToolClient
rather than being called directly — so the loop itself is async (an MCP
session is async under the hood).
"""

import asyncio
import json
import logging
import os
import signal
import time
from contextlib import nullcontext

from .llm_client import chat, LLMError

from .config import AgentConfig
from .intent import extract_intent
from .mcp_client import MCPToolClient
from .session_store import SessionStore

try:
    from . import ui
    _HAS_UI = True
except ImportError:
    _HAS_UI = False

SYSTEM_PROMPT = """You are a coding agent working within a defined project \
directory. You have tools to read, search, write, and edit files, check git \
diffs, and run shell commands.

Rules:
- Prefer edit_file over write_file for existing files — write_file will \
refuse to overwrite unless you pass overwrite=true.
- Before running anything destructive or irreversible, check git_diff or \
read_file first so you understand current state.
- Keep changes minimal and focused on the task.
- When you learn a durable fact about this project (a convention, gotcha, \
build quirk, or stated preference) that would help in a future session, \
call save_memory to persist it. Keep notes short and skip anything already \
obvious from the code itself.
- When the task is fully done, reply with plain text (no tool call) \
summarizing what changed and how to verify it (e.g. which command to run).
"""


def _load_project_memory(project_root: str, memory_path: str) -> str:
    """Read back whatever save_memory has accumulated for this project, so
    it can be folded into the system prompt at the start of a new session."""
    path = os.path.join(project_root, memory_path)
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("omni")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()   # clear() alone drops the reference and leaks the open file
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    return logger


async def _approve(tool_name: str, args: dict, cfg: AgentConfig, client: MCPToolClient, force_approval: bool = False) -> bool:
    if tool_name in cfg.safe_tools:
        return True
    if cfg.auto_approve and not force_approval:
        return True
    # Both paths below read stdin synchronously, which blocks the event loop
    # until answered — deliberate (approvals are serialized anyway), but it
    # does mean a Ctrl+C typed at the prompt isn't acted on until the line is
    # submitted. Treat an interrupt or EOF there as "no".
    try:
        if _HAS_UI:
            return await ui.request_approval(tool_name, args, client)
        print(f"\n--- Approval needed: {tool_name} ---")
        print(json.dumps(args, indent=2)[:2000])
        return input("Proceed? [y/N] ").strip().lower() == "y"
    except (KeyboardInterrupt, EOFError):
        print()
        return False


def _find_json_objects(text: str) -> list:
    """Scan text for top-level {...} objects, tracking string-literal state
    so braces inside quoted content (e.g. code the model is trying to write)
    don't throw off the balance count."""
    objs = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            depth, in_str, esc, j = 0, False, False, i
            while j < n:
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        objs.append(text[i:j + 1])
                        break
                j += 1
            i = j + 1
        else:
            i += 1
    return objs


# How much prose may surround recovered JSON tool calls before the message
# is read as narration rather than as an attempted call (non-whitespace
# characters). A lead-in like "Sure, reading that file now." stays under it.
_MAX_RECOVERY_PROSE_CHARS = 200


def _recover_text_tool_calls(content: str, tool_names: set) -> list:
    """Some models print a tool call as plain-text JSON (`{"name": ...,
    "arguments": {...}}`) instead of using the tool-calling API, which would
    otherwise look like a final answer and silently end the run without the
    tool ever executing. Recover any such calls from `content`.

    Only when the message is essentially nothing *but* those objects: a
    final summary that merely describes a call it already made ("then I sent
    {"name": "write_file", ...}") must not be re-executed, so anything with
    more than _MAX_RECOVERY_PROSE_CHARS of surrounding prose is left alone
    and treated as the plain-text answer it looks like."""
    if not content or "{" not in content:
        return []
    calls, matched = [], []
    for raw in _find_json_objects(content):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name, args = obj.get("name"), obj.get("arguments")
        if name in tool_names and isinstance(args, dict):
            matched.append(raw)
            calls.append({
                "id": f"fallback_{len(calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })
    if not calls:
        return []
    leftover = content
    for raw in matched:
        leftover = leftover.replace(raw, "", 1)
    leftover = leftover.replace("```json", "").replace("```", "")
    if len("".join(leftover.split())) > _MAX_RECOVERY_PROSE_CHARS:
        return []
    return calls


def _ensure_tool_call_ids(tool_calls: list, step: int) -> None:
    """Give every tool call an id, in place, so each result message can name
    the call it answers (see the tool_call_id note in _run_loop). Servers
    normally supply one; a few OpenAI-compatible backends don't."""
    for i, call in enumerate(tool_calls):
        if isinstance(call, dict) and not call.get("id"):
            call["id"] = f"call_{step}_{i}"


def _format_elapsed(seconds: float) -> str:
    """Plain-text mirror of ui._format_elapsed, for the no-rich print()
    fallback path — duplicated rather than imported since this module must
    keep working when ui.py (and rich) isn't installed at all."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _protected_head_len(messages: list) -> int:
    """How many leading messages compaction must never touch: the whole
    leading run of system messages plus the user task that follows them.

    Not a fixed 2 — _run_loop inserts the parsed-intent block as a second
    system message, which pushed the actual task to index 2 and made it
    eligible for summarizing away, exactly the message that must survive."""
    head = 0
    while head < len(messages) and messages[head].get("role") == "system":
        head += 1
    if head < len(messages) and messages[head].get("role") == "user":
        head += 1
    return head


def _trim_history(messages: list, budget: int) -> list:
    """Keep the system + user task message plus the most recent turns
    within a rough character budget. Crude but effective without pulling
    in a tokenizer dependency. Used as a fallback if LLM-based compaction
    (_compact_messages) itself fails."""
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= budget:
        return messages
    head_len = _protected_head_len(messages)
    head, tail = messages[:head_len], messages[head_len:]
    while tail and total > budget:
        removed = tail.pop(0)
        total -= len(str(removed.get("content", "")))
    return head + tail


def _render_for_summary(m: dict, max_len: int = 800) -> str:
    role = m.get("role", "")
    if role == "assistant" and m.get("tool_calls"):
        calls = ", ".join(
            f"{c['function']['name']}({c['function'].get('arguments', '')})"
            for c in m["tool_calls"]
        )
        return f"assistant: called {calls}"
    content = (m.get("content") or "").strip()
    if len(content) > max_len:
        content = content[:max_len] + "…"
    return f"{role}: {content}"


_COMPACT_PROMPT = (
    "You are compacting an in-progress coding-agent session so it can continue "
    "with a smaller context window. Summarize the conversation excerpt below into "
    "a compact briefing for the model that will keep working: what task is being "
    "done, what files were read/written and how, what decisions or constraints "
    "were established, what worked, what failed, and any state the continuation "
    "needs to know. Be concrete (file paths, function names) and terse — this "
    "replaces the raw messages, so omit anything not needed to keep working "
    "correctly. Do not add commentary about the summarization itself."
)


async def _compact_messages(messages: list, model: str, cfg: AgentConfig, logger) -> list:
    """Replace the middle of a long conversation with an LLM-written summary,
    keeping the system + task messages and the most recent `cfg.compact_keep_last`
    messages verbatim. Returns `messages` unchanged if there's nothing worth
    compacting (short history, or no middle to summarize). Falls back to the
    crude drop-oldest trim (_trim_history) if the summarization call fails."""
    keep_last = max(cfg.compact_keep_last, 0)
    head_len = _protected_head_len(messages)
    if len(messages) <= head_len + keep_last:
        return messages

    # keep_last=0 has to slice as "nothing", not messages[-0:] (everything).
    tail = messages[-keep_last:] if keep_last else []
    head, middle = messages[:head_len], messages[head_len:len(messages) - keep_last]
    if not middle:
        return messages

    transcript = "\n".join(_render_for_summary(m) for m in middle)
    start = time.monotonic()
    spinner = ui.thinking(f"Compacting {len(middle)} messages…") if _HAS_UI else nullcontext()
    try:
        with spinner:
            reply = await chat(
                model=model,
                messages=[
                    {"role": "system", "content": _COMPACT_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                base_url=cfg.llm_host, api_key=cfg.llm_api_key, timeout=cfg.llm_timeout_s,
            )
        summary = (reply.get("content") or "").strip()
    except Exception as e:
        logger.info(f"compaction failed, falling back to drop-oldest trim: {e}")
        return _trim_history(messages, cfg.context_char_budget)

    if not summary:
        return _trim_history(messages, cfg.context_char_budget)

    elapsed = time.monotonic() - start
    summary_msg = {
        "role": "system",
        "content": f"# Compacted history ({len(middle)} earlier messages)\n{summary}",
    }
    logger.info(f"compacted {len(middle)} messages into a {len(summary)}-char summary ({elapsed:.1f}s)")
    if _HAS_UI:
        ui.compacted(len(middle), len(summary), elapsed)
    else:
        print(f"Compacted {len(middle)} messages into a {len(summary)}-char summary ({_format_elapsed(elapsed)})")
    return head + [summary_msg] + tail


class CodingAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.logger = _setup_logger(cfg.log_path)
        self.force_approval = False  # set True for the run if intent is high-risk
        self.last_reasoning = ""     # chain of thought from the latest reply, for /reasoning
        self.store = SessionStore(cfg.db_path)
        self.session_id = None  # set by run() to whichever session the last turn used

    async def _call_model(self, messages: list, tool_schemas: list):
        """Call the LLM server with retries for transient errors (connection refused,
        5xx, malformed tool-call output — small local models occasionally
        emit broken JSON)."""
        last_err = None
        start = time.monotonic()
        spinner = ui.thinking() if _HAS_UI else nullcontext()
        with spinner:
            for attempt in range(1, self.cfg.max_retries + 1):
                try:
                    result = await chat(model=self.cfg.model, messages=messages, tools=tool_schemas,
                                         base_url=self.cfg.llm_host, api_key=self.cfg.llm_api_key,
                                         timeout=self.cfg.llm_timeout_s)
                    elapsed = time.monotonic() - start
                    if _HAS_UI:
                        ui.elapsed_note("Responded", elapsed)
                    else:
                        print(f"Responded ({_format_elapsed(elapsed)})")
                    return result
                except LLMError as e:
                    last_err = e
                    self.logger.info(f"model call failed (attempt {attempt}): {e}")
                    if _HAS_UI:
                        spinner.update(f"[bold yellow]Thinking… (retry {attempt}/{self.cfg.max_retries})[/bold yellow]")
                    await asyncio.sleep(min(2 ** attempt, 10))
                except Exception as e:
                    last_err = e
                    self.logger.info(f"unexpected error (attempt {attempt}): {e}")
                    if _HAS_UI:
                        spinner.update(f"[bold yellow]Thinking… (retry {attempt}/{self.cfg.max_retries})[/bold yellow]")
                    await asyncio.sleep(1)
        raise RuntimeError(f"Model call failed after {self.cfg.max_retries} attempts: {last_err}")

    async def compact_history(self, session_id: str) -> str:
        """Manually compact a session's stored history (the /compact REPL
        command). Unlike the automatic compaction inside _run_loop, this runs
        regardless of the char budget and persists the result back to the
        session store, so it takes effect immediately and survives resume.
        Returns a short human-readable message describing what happened."""
        messages = self.store.load_messages(session_id)
        compacted = await _compact_messages(
            messages, self.cfg.compact_model or self.cfg.model, self.cfg, self.logger,
        )
        if len(compacted) >= len(messages):
            return "Nothing to compact — history is already short."
        self.store.replace_messages(session_id, compacted)
        return f"Compacted {len(messages)} messages down to {len(compacted)}."

    async def run(self, task: str = "", resume_session_id: str = None, client: MCPToolClient = None,
                  session_name: str = None, show_banner: bool = True) -> str:
        """Run one turn of the agent loop. If `client` is given (an already
        -open MCPToolClient), it's reused instead of spawning a fresh MCP
        server subprocess — used by the interactive REPL so each turn
        doesn't pay subprocess-startup cost. `self.session_id` is set to
        whichever session this turn ran against, so callers (e.g. the REPL)
        can pass it back in as `resume_session_id` on the next turn.
        `resume_session_id` accepts either a session id or a --session-name.
        `session_name` optionally names a newly-created session. Pass
        `show_banner=False` when a caller (e.g. the REPL) already prints its
        own header and doesn't want one repeated every turn."""
        resuming = resume_session_id is not None

        if resuming:
            session_id = self.store.resolve_session_id(resume_session_id)
            if session_id is None:
                raise ValueError(f"No session found with id or name {resume_session_id!r}")
            messages = self.store.load_messages(session_id)
            persisted = len(messages)  # already in the DB, don't re-write these
            label = task or "[continuing previous task]"
            if _HAS_UI and show_banner:
                ui.banner(f"(resumed {session_id}) {label}", self.cfg.model)
            self.logger.info(f"RESUME session={session_id} TASK: {label}")
            if task:
                messages.append({"role": "user", "content": task})
        else:
            if _HAS_UI and show_banner:
                ui.banner(task, self.cfg.model)
            session_id = self.store.create_session(self.cfg.project_root, self.cfg.model, task, name=session_name)
            # cfg.system_prompt replaces the built-in one; empty falls back
            # to it. Whatever is chosen here is stored as the session's first
            # message, so a resumed session keeps the prompt it started with
            # rather than silently adopting a new flag value.
            system_content = self.cfg.system_prompt.strip() or SYSTEM_PROMPT
            memory_text = _load_project_memory(self.cfg.project_root, self.cfg.memory_path)
            if memory_text:
                system_content += "\n\n# Project memory (persisted from previous sessions)\n" + memory_text
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": task},
            ]
            persisted = 0
            self.logger.info(f"TASK: {task} (session={session_id})")

        self.session_id = session_id

        try:
            if client is not None:
                return await self._run_loop(task, session_id, messages, persisted, resuming, client)
            async with MCPToolClient(self.cfg.project_root, mcp_config_path=self.cfg.mcp_config_path or None,
                                      extra_servers=self.cfg.mcp_servers or None,
                                      embedding_model=self.cfg.embedding_model or None,
                                      llm_host=self.cfg.llm_host or None,
                                      llm_api_key=self.cfg.llm_api_key or None,
                                      mcp_log_path=self.cfg.mcp_log_path,
                                      builtin_env=self.cfg.tool_server_env()) as owned_client:
                return await self._run_loop(task, session_id, messages, persisted, resuming, owned_client)
        except asyncio.CancelledError:
            # Ctrl+C during a turn. CancelledError derives from BaseException,
            # not Exception, so without this the session would keep its
            # initial "running" status forever — still listed as in-flight by
            # /sessions and --list-sessions long after the process exited.
            self.store.finish_session(session_id, "interrupted", "Interrupted by user (Ctrl+C).")
            raise
        except Exception as e:
            self.store.finish_session(session_id, "error", str(e))
            raise

    async def _run_loop(self, task: str, session_id: str, messages: list, persisted: int,
                         resuming: bool, client: MCPToolClient) -> str:
        tool_schemas = await client.list_llm_tools()
        tool_names = {t["function"]["name"] for t in tool_schemas}

        # Every new instruction gets parsed, including later turns of an
        # interactive session (which all arrive with resuming=True). Doing
        # this only on turn 1 meant a "delete the migrations and force-push"
        # typed at turn 5 was never risk-classified, so --auto-approve sailed
        # straight past the one gate that exists for it.
        if task and self.cfg.parse_intent:
            # Re-evaluated per instruction rather than latched for the
            # process: a high-risk turn must not leave every later turn
            # prompting, and a low-risk turn 1 must not disarm turn 5.
            self.force_approval = False
            intent_model = self.cfg.intent_model or self.cfg.model
            spinner = ui.thinking("Parsing intent…") if _HAS_UI else nullcontext()
            with spinner:
                intent = await extract_intent(task, intent_model, self.cfg.max_retries, self.logger,
                                               base_url=self.cfg.llm_host, api_key=self.cfg.llm_api_key,
                                               timeout=self.cfg.llm_timeout_s)

            existing = {f: await client.file_exists(f) for f in intent.target_files}
            context_block = intent.as_context_block(existing)
            # Immediately before the instruction it describes — index 1 for a
            # fresh session, just before the freshly appended task for a
            # resumed one. Anything already persisted stays put, so the
            # messages[persisted:] write-out below keeps DB order intact.
            insert_at = max(
                (i for i, m in enumerate(messages) if m.get("role") == "user"),
                default=len(messages),
            )
            messages.insert(insert_at, {"role": "system", "content": context_block})

            if _HAS_UI:
                ui.intent_panel(intent, existing)
            else:
                print(f"\n{context_block}\n")

            if intent.risk_level == "high":
                self.force_approval = True
                warning = "High-risk intent detected — approval required for all write/shell actions this run, even with --auto-approve."
                if _HAS_UI:
                    ui.high_risk_warning()
                else:
                    print(f"⚠️  {warning}")
                self.logger.info(warning)

        for m in messages[persisted:]:
            self.store.append_message(session_id, persisted, m)
            persisted += 1

        for step in range(1, self.cfg.max_steps + 1):
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            if total_chars > self.cfg.context_char_budget:
                compacted = await _compact_messages(
                    messages, self.cfg.compact_model or self.cfg.model, self.cfg, self.logger,
                )
                if compacted is not messages:
                    # Mirror the compaction into the store, the way the manual
                    # /compact command does. Without this the DB keeps the full
                    # pre-compaction history, so resuming this session reloads
                    # everything that was just summarized away and blows the
                    # budget again on the first turn.
                    messages = compacted
                    self.store.replace_messages(session_id, messages)
                    persisted = len(messages)
            msg = await self._call_model(messages, tool_schemas)

            tool_calls = msg.get("tool_calls")
            recovered_from_text = False
            if not tool_calls:
                recovered = _recover_text_tool_calls(msg.get("content", ""), tool_names)
                if recovered:
                    msg["tool_calls"] = recovered
                    tool_calls = recovered
                    recovered_from_text = True
                    self.logger.info(
                        f"[step {step}] model printed tool call as plain text; "
                        f"recovered {len(recovered)} call(s) via fallback parsing"
                    )

            if tool_calls:
                _ensure_tool_call_ids(tool_calls, step)

            messages.append(msg)
            self.store.append_message(session_id, persisted, msg)
            persisted += 1

            # Chain of thought, when the server sends it as its own field
            # (llama.cpp, vLLM's reasoning parsers, DeepSeek). Kept collapsed:
            # it typically dwarfs the answer, so the transcript gets one dim
            # line and /reasoning opens it. Skipped when the reasoning IS the
            # content — llm_client falls back to that field for a reply that
            # carried nothing else, and echoing it above itself prints it twice.
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
            if reasoning and reasoning != (msg.get("content") or "").strip():
                self.last_reasoning = reasoning
                if _HAS_UI:
                    ui.reasoning_note(reasoning)

            # What the model said alongside its tool calls. Shown before the
            # calls run, so the transcript above the prompt reads narration
            # then actions. Skipped when the "content" IS the tool call
            # (recovered_from_text) — printing raw JSON adds nothing — and on
            # the final turn, where cli.py renders it as the result panel.
            interim = "" if recovered_from_text else (msg.get("content") or "").strip()
            if tool_calls and interim:
                if _HAS_UI:
                    ui.assistant_message(interim)
                else:
                    print(f"\n{interim}")

            if not tool_calls:
                final = msg.get("content", "")
                self.logger.info(f"DONE: {final}")
                self.store.finish_session(session_id, "done", final)
                return final

            # Parse every call's arguments up front. The whole step (all
            # tool calls the model made this turn — often several at once)
            # is displayed as a single unit once execution finishes, instead
            # of announcing + reporting each call separately, and the calls
            # themselves run concurrently rather than one after another.
            calls = []
            for call in tool_calls:
                name = call["function"]["name"]
                raw_args = call["function"]["arguments"]
                args = raw_args
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = None
                calls.append({"name": name, "args": args, "raw": raw_args, "id": call.get("id")})

            for c in calls:
                if c["args"] is None:
                    c["result"] = f"ERROR: model sent malformed arguments: {c['raw']!r}"

            runnable = [c for c in calls if c["args"] is not None]

            # Approval prompts are interactive, so they're resolved one at a
            # time in call order; the tool calls that get approved then run
            # concurrently below instead of one after another.
            for c in runnable:
                c["approved"] = await _approve(c["name"], c["args"], self.cfg, client, self.force_approval)

            async def _execute(c):
                start = time.monotonic()
                try:
                    if not c["approved"]:
                        return "Denied by human reviewer. Choose a different approach."
                    return await client.call_tool(c["name"], c["args"])
                except Exception as e:
                    return f"ERROR: {c['name']} raised: {e}"
                finally:
                    c["duration"] = time.monotonic() - start

            if runnable:
                names = ", ".join(c["name"] for c in runnable)
                spinner = ui.thinking(f"Running {names}…") if _HAS_UI else nullcontext()
                with spinner:
                    # Ctrl+C here cancels only the tool call(s) currently
                    # running, not the whole turn — swap in a handler that
                    # targets just this step's task(s), restoring whatever
                    # was active before (cli.py's "cancel the whole turn"
                    # handler, during every other phase) once done.
                    if all(c["name"] in self.cfg.safe_tools for c in runnable):
                        # Every call is read-only, so they all run at once —
                        # Ctrl+C cancels whichever of them haven't finished yet.
                        tasks = [asyncio.ensure_future(_execute(c)) for c in runnable]
                        previous_sigint = signal.signal(
                            signal.SIGINT, lambda *_: [t.cancel() for t in tasks if not t.done()]
                        )
                        try:
                            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
                        finally:
                            signal.signal(signal.SIGINT, previous_sigint)
                    else:
                        # Sequential (see the safe_tools check above for why):
                        # only the one call actually in flight is cancelable
                        # — each task is created and awaited in turn, not all
                        # up front. Once one is cancelled, the rest of this
                        # step's calls are skipped rather than run anyway —
                        # Ctrl+C means "stop", not "skip just this one and
                        # keep mutating state with whatever comes next."
                        current = {}
                        previous_sigint = signal.signal(
                            signal.SIGINT,
                            lambda *_: current["task"].cancel() if current.get("task") and not current["task"].done() else None,
                        )
                        try:
                            raw_results = []
                            cancelled = False
                            for c in runnable:
                                if cancelled:
                                    raw_results.append(asyncio.CancelledError())
                                    continue
                                current["task"] = asyncio.ensure_future(_execute(c))
                                try:
                                    raw_results.append(await current["task"])
                                except asyncio.CancelledError:
                                    raw_results.append(asyncio.CancelledError())
                                    cancelled = True
                        finally:
                            signal.signal(signal.SIGINT, previous_sigint)

                results = [
                    "ERROR: cancelled by user (Ctrl+C)." if isinstance(r, asyncio.CancelledError) else r
                    for r in raw_results
                ]
                for c, result in zip(runnable, results):
                    c["result"] = result

                if any(c["name"] == "search_tools" and not str(c["result"]).startswith("ERROR") for c in runnable):
                    # Deferred-loading MCP tools just got revealed — refresh
                    # the schemas handed to the model so it can call them.
                    tool_schemas = await client.list_llm_tools()
                    tool_names = {t["function"]["name"] for t in tool_schemas}

            for c in calls:
                c["ok"] = (
                    c["args"] is not None
                    and not str(c["result"]).startswith("ERROR")
                    and c["result"] != "Denied by human reviewer. Choose a different approach."
                )

            if _HAS_UI:
                ui.step_display(calls)
            else:
                for c in calls:
                    print(f"[step {step}] {c['name']}({c['args']}) -> {str(c['result'])[:200]}")

            for c in calls:
                self.logger.info(f"[step {step}] {c['name']}({c['args']}) -> {str(c['result'])[:500]}")
                # tool_call_id pairs this result with the call it answers.
                # Required by the OpenAI chat-completions schema for role
                # "tool": Ollama tolerates its absence, but a strict server
                # (OpenAI, vLLM, OpenRouter) rejects the next request with a
                # 400 the moment a run reaches its second step.
                tool_msg = {"role": "tool", "content": str(c["result"])}
                if c.get("id"):
                    tool_msg["tool_call_id"] = c["id"]
                messages.append(tool_msg)
                self.store.append_message(session_id, persisted, messages[-1])
                persisted += 1

        msg = "Max steps reached without completion. Check the log for progress."
        self.logger.info(msg)
        self.store.finish_session(session_id, "max_steps", msg)
        return msg
