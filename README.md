# 🐙 Omni Coder

## ⚙️ Setup
```bash
ollama pull qwen3-coder:30b   # example: pulling the default model via Ollama
pip install -e .
```
No vendor SDK required — the agent talks to an OpenAI-compatible
chat-completions endpoint (`/api/v1/chat/completions`) directly over HTTP via
`httpx`. This works against Ollama, vLLM, LM Studio, or any other
OpenAI-compatible server/gateway (e.g. Open WebUI) — point `--llm-host` at
whichever one you're running.

## ▶️ Run
```bash
omni "Add type hints to utils.py, then run the test suite" \
    --project-root ./myrepo
```
Equivalent alternative: `python -m omni "..." --project-root ./myrepo`.

Add `--auto-approve` to skip confirmation prompts (only in an already-isolated
environment, e.g. a container you're fine getting wiped). Add `--max-steps N`
to change the default cap of 100 agent-loop iterations. Run `omni --help`
for the full option list — it's a Typer app, so `--help` is auto-generated and
kept in sync with the code.

Omit the task string to drop into an interactive session instead of a
one-shot run — see [Session management](#session-management) below.

## 🏭 What makes this "production grade" vs. the first draft

| Concern | First draft | This version |
|---|---|---|
| Editing existing files | Only full overwrite via `write_file` | `edit_file` does exact unique-match replace + shows a unified diff, `write_file` refuses to clobber existing files |
| Path safety | None — agent could read/write anywhere | Every path resolved and checked against `project_root`; escapes raise `PathScopeError` |
| Shell safety | Ran anything, unbounded | Denylist for destructive patterns (`rm -rf /`, `sudo`, fork bombs, etc.), timeout, output truncation |
| Human oversight | None | Write/edit/shell calls pause for approval unless the tool is in `safe_tools` or `auto_approve=True` |
| Model reliability | Assumed clean tool-call JSON | Retries with backoff on API errors; malformed tool-call args are caught and reported back to the model instead of crashing |
| Context window | Unbounded growth | Once the conversation exceeds `--context-char-budget` (default 200k chars), it's compacted via an LLM-written summary instead of growing forever |
| Observability | `print()` only | Structured log file (`agent_run.log`) recording every model call, tool call, args, and result — plus a separate `mcp_servers.log` for MCP server stderr, and a `/mcp` command showing live connection status |
| Config | Hardcoded constants | `AgentConfig` dataclass — one place to tune model, project root, limits, policy |
| Sessions | Each run started from a blank conversation | Every message is persisted to SQLite (`session_store.py`); resume by id or name, or run interactively |
| Codebase search | `grep` piped through a subprocess | Pure-Python `search_files` (regex + glob filter, skips `.git`/`node_modules`/etc.) and a `glob_files` tool for pattern-based file discovery |
| Git integration | Only `git_diff` (read-only) | Full toolset — `git_status`, `git_log`, `git_show`, `git_branch`, `git_fetch` (read-only, auto-approved) plus `git_add`, `git_commit`, `git_pull`, `git_push` (approval-gated, same as any other write) |
| Long-term memory | Every session starts blank | `save_memory` tool appends durable notes (conventions, gotchas, preferences) to a project-local `agent_memory.md`, auto-injected into the system prompt at the start of every new session |
| Interactive robustness | Ctrl+C anywhere killed the whole process | Ctrl+C during a running turn cancels just that turn (via a scoped SIGINT handler + `asyncio.Task.cancel()`) — the session and MCP connection stay alive so you can keep going |

## ⚠️ Still recommended before real production use

1. **Run it in a container**, not on your host. The path-scope check and shell
   denylist reduce risk but are not a substitute for OS-level isolation —
   treat `run_shell` as "can execute arbitrary code" and contain the blast
   radius accordingly (Docker, gVisor, a disposable VM).
2. **Version control everything.** Require the project root to be a git repo
   and commit before each run, so any agent change is a reviewable diff you
   can revert.
3. **Rate/step limits per user** if this is exposed to a team, not just you.
4. **Swap the char-based context trimming for a real tokenizer** if you hit
   context issues in practice — it's a rough approximation.
5. **Add tests for the tools module** (`tools.py`) in your CI — the
   path-scope check is the one thing you really don't want to regress silently.
6. Qwen3-Coder's native tool-calling is solid but not perfect at this size —
   watch the log for `BAD ARGS` entries; if they're frequent, consider a
   larger quant or `qwen2.5-coder:32b` (dense, less agentic-tuned but very
   reliable on straightforward edits).

## 🧭 Intent parsing

Before the agent takes any action, the raw task string is parsed by the model
(in strict JSON mode, no tools) into structured intent:

```json
{
  "task_type": "bugfix",
  "summary": "Fix add() which subtracts instead of adding",
  "target_files": ["math_utils.py"],
  "constraints": [],
  "risk_level": "low"
}
```

This gets injected into the conversation as a system message (with each
target file tagged `exists` or `new` within the project root), so the model
starts with grounded structure instead of just the raw sentence. Two things
follow from this automatically:

- **High-risk tasks force approval**, even if you ran with `--auto-approve`.
  Detected via `risk_level: "high"` (deletion, deploys, migrations, etc.).
- **Malformed or failed parsing degrades gracefully** — after retries, it
  falls back to `task_type: "other"` with `confident=False` logged, and the
  agent still runs on the raw task text rather than blocking.

Skip it with `--skip-intent-parsing` if you want lower latency on simple
tasks, or point it at a smaller/faster model with `--intent-model`.

## 🧠 Project memory

The agent can remember durable facts about a project across separate runs —
conventions, gotchas, stated preferences — via a `save_memory` tool it calls
on its own (the system prompt tells it when: durable and non-obvious, not
task-specific status). Notes are appended as timestamped bullets to a
project-local file, `agent_memory.md` by default (`--project-root`-relative,
not CWD-relative), e.g.:

```
- [2026-07-24] This repo uses pytest, not unittest.
- [2026-07-24] Config is loaded from .env, never hardcode secrets.
```

Whatever's in that file is read back and folded into the system prompt at the
start of every new (non-resumed) session — no flag needed, no separate
"recall" step. `save_memory` is in `safe_tools` by default (auto-approved):
it only ever appends a line to a bookkeeping file, never touches real source,
so it doesn't warrant a confirmation prompt like `write_file`/`edit_file` do.
Resumed sessions keep whatever memory was baked in when they originally
started rather than re-reading the file live — kept intentionally simple.

## 💾 Session management

Every message in the conversation — system, user, assistant, tool results —
is persisted to a SQLite file (`agent_sessions.db` by default, `--db-path` to
change it) as the run happens, via `session_store.py`. A session is done the
moment it's created; nothing extra to opt into.

**Resume a previous run:**
```bash
omni "Add type hints to utils.py" --session-name utils-typing
# ...later, in the same or a different terminal...
omni --resume utils-typing "Also add docstrings"
```
`--resume` accepts either the session id it printed at the end of a run, or
the `--session-name` you gave it. `--session-name` is optional — without it
you just get an 8-character id. When you resume, the prior conversation is
printed before the run continues (assistant replies rendered the same
Markdown-panel way they looked the first time), so it's visibly clear that
context carried over rather than just trusting it happened in the background.

**Browse saved sessions:**
```bash
omni --list-sessions
```
Shows id, name, status (`running` / `done` / `max_steps` / `error`), last
updated time, model, and the original task for each session.

**Delete a session:**
```bash
omni --delete-session utils-typing
```
Removes the session and its full message history. Also available as
`/delete <id-or-name>` from inside interactive mode.

**Interactive mode** — omit the task argument entirely to get a REPL instead
of a one-shot run:
```bash
omni --project-root ./myrepo              # fresh session, prompts for input
omni --resume utils-typing                 # resumes and prompts for input
```
Type a task and press enter to run it; the conversation (and the MCP tool
connection) stays alive between turns, so follow-ups don't pay the cost of
re-parsing intent or re-spawning the tool server. Type `/` at the prompt to
pop a completion menu of every available command — static ones below, plus
one `/model <name>` entry per model the LLM server reports (best-effort;
skipped if it doesn't expose `/v1/models`) and one `/server:prompt` entry per
MCP prompt exposed by a connected server. Special inputs:
- `/sessions` — list saved sessions without leaving the REPL
- `/mcp` — table of every configured MCP server (built-in + custom), each
  with a ✅/❌ connected status, how long it's been connected, tool count,
  and command/URL (or the connection error, for a ❌ one). A custom server
  failing to connect no longer aborts startup — it just shows ❌ here
  instead of the whole session refusing to start
- `/mcp restart <name>` — reconnect one server (`/mcp restart all` for every
  one) after editing its code, without leaving the REPL. Also how you retry
  a ❌ server once you've fixed it. It re-reads that server's spec from the
  settings file too, so edits to its `command`/`args`/`env`/`headers` are
  picked up as well; the next turn sees its new tool list, prompts, and
  resources. Works on the `built-in` server too, for when you change
  `tools.py`
- `/model` — opens an interactive picker (↑/↓ to move, Enter to select, Esc to
  cancel) of models available on the LLM server, defaulting to the current one
- `/model <name>` — switch the active model directly, without the picker
- `/resources` — list resources published by connected MCP servers (the MCP
  "Resources" capability — readable context addressed by URI);
  `/resources <uri>` prints one. See
  [MCP resources](#-mcp-resources) — the model can read these too
- `/server:prompt [param1] [param2] ...` — resolve an MCP prompt template (the
  MCP "Prompts" capability — user-invocable templates a server exposes,
  distinct from tools) exposed by a connected server, and run it as the task.
  Arguments are positional, matched in order against the prompt's declared
  argument list (quote a value to include spaces, e.g. `/docs:search "foo bar"`)
- `/delete <id-or-name>` — delete a saved session without leaving the REPL
- `/compact` — summarize the current session's history down to the system
  prompt, original task, and most recent messages (`--compact-keep-last`,
  default 20), replacing everything older with an LLM-written briefing.
  Persists immediately, so the shrunk history is what future turns (and
  `--resume`) load. History is also compacted automatically mid-run
  whenever it exceeds `--context-char-budget` (default 200,000 characters,
  not tokens — a rough proxy); that automatic pass only affects the model's
  working context and doesn't rewrite saved history. Use `--compact-model`
  to run the summarization call itself through a smaller/faster model
  than `--model` (same idea as `--intent-model`).
- **Ctrl-C while a turn is running** — interrupts just that turn (cancels
  whatever model or tool call is in flight) and drops you back at the
  prompt; the session and MCP connection stay alive, so you can keep
  chatting or ask the agent to pick up where it left off. Progress up to
  the last completed step is already saved.
- `/exit` or `/quit` (or Ctrl-D, or Ctrl-C at an idle prompt) — leave

A spinner shows while waiting on the model (initial intent parsing and every
turn), including a live retry counter if a call fails transiently and gets
retried — so a slow or cold-loading model doesn't look like it's hung.

## 🎨 Terminal UI

`ui.py` renders everything through [rich](https://github.com/Textualize/rich):
banner + parsed intent as a panel, each step with a colored ✓/✗, approval
prompts that show the actual diff/command *before* you approve — not just the
raw args — and the final response rendered as Markdown (headers, lists, code
blocks) rather than literal text.

- **`edit_file` diffs** (both the approval preview and the post-edit result)
  render through a custom GitHub-style diff view — a line-number gutter,
  removed lines in bold red, added lines in bold green — instead of generic
  pygments diff coloring. `write_file`'s overwrite preview uses the same
  renderer; new-file content is syntax-highlighted by extension instead.
- **`search_files` and `read_file` step output is summarized, not dumped** —
  you see `"3 matches found"` or `"42 lines (1180 chars)"` rather than the
  actual matched lines or file content scrolling past. The model still gets
  the full text either way; this only changes what's echoed to the terminal.

If `rich` isn't installed, `agent.py` and `cli.py` both detect the missing
import and fall back to plain `print()` — nothing breaks, it just looks
like the original CLI.

## 🔌 Tools as an MCP server

The tools live behind a real MCP server (`mcp_server.py`), not inline in the
agent. The agent is an MCP *client* — it spawns the server as a subprocess
(stdio transport, launched as `python -m omni.mcp_server` so its
relative imports resolve) scoped to `--project-root`, fetches the tool list,
converts it to OpenAI's function-calling schema, and calls tools through the
MCP session instead of Python function calls directly.

```
+------------------------------------------+
|        cli.py  (Typer CLI / REPL)        |
+------------------------------------------+
                     |
                     v
+------------------------------------------+
|                 agent.py                 |
|    call model, parse intent, approve,    |
|      execute tools, persist, repeat      |
+------------------------------------------+
                     |
                     v
+------------------------------------------+
|       llm_client.py (model calls)        |
|    session_store.py (SQLite history)     |
+------------------------------------------+
                     |
                     v
+------------------------------------------+
|              mcp_client.py               |
|     built-in + custom servers merged     |
|      into one namespaced tool list.      |
|  Servers marked "defer" hold their tools |
| back; a synthesized search_tools tool    |
|   reveals matches on demand (below)      |
+------------------------------------------+
                     |
       stdio / SSE / streamable-http
                     v
+------------------------------------------+
|  mcp_server.py  /  custom MCP server(s)  |
+------------------------------------------+
                     |
                     v
+------------------------------------------+
|                 tools.py                 |
|  read / write / edit / search / shell,   |
|       each scoped to project_root        |
+------------------------------------------+
```

Resolving a `search_tools` call (inside `mcp_client.py`, see
[Deferred tool loading & search_tools](#deferred-tool-loading--search_tools)):

```
search_tools(query) for a "defer"-registered server's hidden tools
                     |
                     v
              embedding_model?
   |
   |-- "nomic-local" (default) --> nomic package, on-device,
   |                                no server (install below)
   |-- <remote-model-name>     --> llm_client.embed() -> your LLM server
   |-- ""                      --> skip straight to keyword match
   |
   v
cosine-rank matches, above threshold -> reveal
(embedding backend missing/erroring at any point falls back to
 plain keyword substring matching automatically, per call)
```

What this buys you:
- **Any MCP client can use the same tools** — Claude Desktop, another agent
  framework, a different model entirely — all sharing the identical
  path-scope, denylist, and diff-preview logic in `tools.py`.
- **The server is independently runnable and testable**:
  ```bash
  AGENT_PROJECT_ROOT=/path/to/repo python -m omni.mcp_server
  ```
- Internal tools (`_preview_edit`, `_preview_write`, `_file_exists`) are
  underscore-prefixed and filtered out of what's shown to the LLM in
  `list_llm_tools()` — the agent still calls them directly for approval
  previews and intent validation, the model never sees them.

The agent loop is `async` end-to-end (an MCP session requires it); `cli.py`
runs it via `asyncio.run()`.

## 🧩 Custom MCP servers

The built-in tools aren't the ceiling — any MCP server, local (stdio) or
remote (SSE / Streamable HTTP), can be added, and its tools show up to the
model automatically, in the same `tools` list the built-ins use, with no
other wiring needed. They're namespaced as `<server_name>__<tool_name>` so
they can't collide with the built-ins or each other, and go through the same
human-approval flow as every other tool unless added to `safe_tools` or run
with `--auto-approve`.

A custom server failing to connect doesn't take down the whole session —
only the built-in server (which provides the core file/shell tools) is
fatal if it can't start; every other server's connection failure is just
recorded and shown as ❌ in `/mcp` (see [Session management](#session-management)),
so the rest of the session still starts normally. Every stdio-transport
server's stderr — built-in and custom alike — is redirected to
`--mcp-log-path` (default `mcp_servers.log`) instead of the terminal, so a
chatty or crashing server's raw debug output doesn't interleave with the
Rich UI; check that file (not the terminal) when a custom server misbehaves.

There are three ways to add one, and all three accept either a **local
command** (spawned over stdio, like the built-in server) or a **remote
URL** (SSE or Streamable HTTP):

**Register one permanently** — available on every future run, in any
project, with zero flags from then on:
```bash
omni --add-mcp-server "weather=python -m weather_mcp_server"     # local, stdio
omni --add-mcp-server "weather=https://example.com/mcp/sse"      # remote, SSE
omni "what's the forecast?"   # picked up automatically
```
Saved under the `mcpServers` key of `~/.omni-coder/omni-coder-settings.json`
and auto-loaded whenever `--mcp-config` isn't explicitly passed. Any other
top-level key in that file is left untouched when servers are registered or
removed, so it can hold unrelated settings too. (A pre-existing
`~/.omni-coder/mcp.json` from before this rename is migrated automatically
on first use.) Manage the registry with:
```bash
omni --list-mcp-servers
omni --remove-mcp-server weather
```

**Add one for a single run** instead, without saving it:
```bash
omni --mcp-server "weather=python -m weather_mcp_server" "task"
```
Repeatable for multiple servers.

**Or load several from a config file** (same shape Claude Desktop uses,
handy when servers need env vars/auth headers or there are a lot of them):
```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["-m", "weather_mcp_server"],
      "env": { "WEATHER_API_KEY": "$WEATHER_API_KEY" }
    },
    "docs": {
      "url": "https://example.com/mcp/sse",
      "transport": "sse",
      "headers": { "Authorization": "Bearer ${DOCS_TOKEN}" }
    }
  }
}
```
```bash
omni --mcp-config ./mcp.json "task"
```

### Secrets: bearer tokens and env vars

Both `headers` and `env` values support `$VAR` / `${VAR}` references,
resolved from your environment when the server is connected — so the file
stores only the *name* of the secret, never the secret itself, and stays
safe to commit. A referenced variable that isn't set is reported as a clear
error naming it, rather than silently sending the literal `$VAR` as your
credential and failing later as a puzzling 401.

For a remote server, the compact CLI format takes the token directly via a
`,bearer=<token>` suffix, which expands to an `Authorization: Bearer`
header:
```bash
# reference an env var — nothing secret in shell history or on disk
export DOCS_TOKEN="sk-..."
omni --add-mcp-server 'docs=https://example.com/mcp/sse,bearer=$DOCS_TOKEN'

# combine with other suffixes, in either order
omni --mcp-server 'docs=https://example.com/mcp,streamable_http,bearer=$DOCS_TOKEN,defer' "task"
```
Note the **single quotes** — they're what keep `$DOCS_TOKEN` intact for this
agent to resolve later. In double quotes your shell would expand it first,
so `--add-mcp-server` would persist the real token to the settings file in
plaintext, which is exactly what the `$VAR` form avoids. (Pasting a literal
token instead of a reference works, with that same caveat.)

`,bearer=` is rejected for local (stdio) servers, which have no request
headers — pass secrets to those through `env` instead.

All three sources can be combined; `--mcp-server` wins over `--mcp-config`
on a name clash, and an explicit `--mcp-config` wins over the auto-loaded
global registry.

### Iterating on a server you're writing

Editing an MCP server used to mean restarting the whole session to pick up
the change. From the REPL, reconnect just that server instead:
```
❯ /mcp restart docs          # after editing docs-server.js
Restarted MCP server 'docs' (7 tools).

❯ /mcp restart all           # every server, built-in included
```
The old subprocess is stopped before the replacement starts (no orphans),
the server's spec is re-read from the settings file so `command`/`args`/
`env`/`headers` edits apply too, and its tools, prompts, and resources are
re-listed — the next turn sees the new set. Other servers keep their
existing connections. A server that's currently broken reports the failure
in `/mcp` instead of taking the session down, so you can fix it and
`/mcp restart` again.

**Local commands vs. remote URLs:**
- Anything after `name=` that starts with `http://` or `https://` is treated
  as a remote server — SSE by default, or Streamable HTTP if you append
  `,streamable_http` (e.g. `"weather=https://example.com/mcp,streamable_http"`).
  In the JSON config format, set `"url"` (and optionally `"transport"`,
  `"headers"` for auth) instead of `"command"`.
- Anything else is treated as a local command spawned over stdio — it
  doesn't need to be `-m`-invokable; a standalone script works too, e.g.
  `"myserver=python C:/absolute/path/to/mcp_server.py"`. Use an **absolute
  path** for a script file — it's resolved relative to wherever you happen
  to run `omni` from (not `--project-root`), so a relative path
  breaks the moment you run the command from a different directory.
- The compact `--mcp-server`/`--add-mcp-server` string format covers bearer
  auth via `,bearer=<token>` (see above). For any *other* header, or to set
  `env` on a local server, use a JSON config file.

## 📚 MCP resources

Besides tools (callable) and prompts (user-invocable templates), an MCP
server can publish **resources** — readable context addressed by URI: coding
standards, an API schema, a changelog, records from a system the agent
otherwise can't see. Any connected server's resources are picked up
automatically; nothing to configure.

**From the REPL**, `/resources` lists what's available and `/resources <uri>`
prints one (tab-completion is populated with the live URIs):
```
❯ /resources
                                   MCP Resources
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ uri                         ┃ server ┃ type             ┃ description           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ file:///coding-standards.md │ docs   │ text/markdown    │ Team coding standards │
│ file:///api-spec.json       │ docs   │ application/json │ Public API schema     │
│ file:///logs/{date}.log     │ docs   │ template         │ Daily log             │
└─────────────────────────────┴────────┴──────────────────┴───────────────────────┘

❯ /resources file:///api-spec.json
╭────── 📖 file:///api-spec.json ──────╮
│ {"endpoints": ["/users", "/orders"]} │
╰──────────────────────────────────────╯
```

**The model can read them too.** When at least one connected server publishes
a resource, two extra read-only tools appear in its toolset — `list_resources`
and `read_resource(uri)` — both auto-approved, since neither can modify
anything. The known URIs are inlined into `read_resource`'s description, so
for a handful of resources the model can go straight to reading one without
a discovery round trip. Just refer to them in a task:
```bash
omni "read the coding standards resource, then fix utils.py to match"
```
`read_resource` is deliberately distinct from `read_file`: the former reads
MCP-published context by URI, the latter reads a path inside
`--project-root`.

Notes:
- **Templates** (parameterized URIs like `file:///logs/{date}.log`) are listed
  and marked `template`, but can't be read until the placeholders are filled
  in — they're excluded from what's offered to the model.
- **Binary resources** aren't dumped into context as base64; they read back as
  a short `[binary image/png, 108 bytes — not shown]` marker.
- If two servers publish the **same URI**, the first-connected one wins and
  the other is noted in the listing.

## 🔍 Deferred tool loading & search_tools

A custom MCP server with a lot of tools (or ones rarely needed) can be
registered with `"defer": true` instead of loading eagerly. Its tools are
held back from the model's default tool list entirely — instead the model
gets one extra tool, `search_tools`, which it calls with a keyword or
free-text query to load matching tools on demand. This keeps unused tool
schemas out of context on every turn, the same trade-off this harness's own
tool search makes for its own rarely-used tools.

Mark a server as deferred the same three ways you'd register one:
```bash
# permanently, via the global registry
omni --add-mcp-server "docs=node docs-server.js" --defer

# for a single run, via the compact spec (",defer" suffix works for
# stdio and remote/URL specs alike)
omni --mcp-server "docs=node docs-server.js,defer" "task"
omni --mcp-server "docs=https://example.com/mcp,streamable_http,defer" "task"
```
or in a `--mcp-config` JSON file, add `"defer": true` to that server's entry:
```json
{
  "mcpServers": {
    "docs": { "command": "node", "args": ["docs-server.js"], "defer": true }
  }
}
```
`--list-mcp-servers` marks deferred entries with `[defer]`.

Once a query reveals a tool, it stays available for the rest of that run —
`search_tools` never needs to be called twice for the same tool. That
revealed state lives only in memory for the current run, though: resuming a
session later (`--resume`) starts every deferred server's tools hidden
again. An empty query reveals everything still hidden at once.

**Matching is semantic by default**, ranked by cosine similarity between the
query and each hidden tool's name + description, so a query doesn't need to
share literal keywords with the tool it's after:
```bash
omni --embedding-model "" "task"                 # disable, plain keyword match only
omni --embedding-model mxbai-embed-large "task"  # use a remote OpenAI-compatible embedding model instead
```
- **Default (`nomic-local`)** — runs on-device via the `nomic` package, no
  server or API key involved. It's declared as this project's own
  `local-embeddings` extra (`pyproject.toml`), so install it with:
  ```bash
  pip install "omni-coder[local-embeddings]"   # from PyPI
  pip install -e ".[local-embeddings]"                  # from a local checkout
  ```
  (optional, not a hard dependency — pulls in `nomic[local]`, and the model
  itself downloads on first use). Without it installed, `search_tools`
  automatically falls back to plain keyword matching and says so in its
  result.
- **A remote model name** (e.g. `mxbai-embed-large`) instead embeds via the
  same `--llm-host`/`--llm-api-key` this agent already talks to for
  chat — pull it there first (`ollama pull mxbai-embed-large`).
- **`""`** disables semantic ranking outright; `search_tools` then requires
  every word in the query to literally appear in a tool's name/description.
- Any embedding failure (dependency missing, model not pulled, network
  error) falls back to keyword matching for that call rather than erroring
  out — `search_tools` still answers, just less precisely.

## 📁 Files
All modules live under `omni/`:
- `config.py` — all tunables in one dataclass
- `intent.py` — parses the freeform task into structured intent (task_type, target_files, constraints, risk_level)
- `tools.py` — tool implementations, each scoped to `project_root` (used by `mcp_server.py`, not called directly by the agent anymore) — read/write/edit/search/shell, a full git toolset, and `save_memory`
- `mcp_server.py` — MCP server exposing those tools over stdio
- `mcp_client.py` — async MCP client the agent uses to reach the server; also merges in any custom MCP servers, and implements deferred tool loading + the `search_tools` tool (semantic ranking via `nomic[local]` or a remote embedding model, falling back to keyword matching)
- `llm_client.py` — raw `httpx` client for the model's OpenAI-compatible chat-completions endpoint (`chat()`) and embeddings endpoint (`embed()`, used by `mcp_client.py`'s `search_tools`) — no vendor SDK dependency, works against any OpenAI-compatible server
- `session_store.py` — SQLite persistence for sessions and their full message history (resume/list/interactive mode)
- `ui.py` — rich terminal rendering (diffs, panels, approval prompts, session tables) — purely presentational
- `agent.py` — the loop: parse intent, call model, approve, execute via MCP, persist, repeat
- `cli.py` — command-line entry point (Typer — `omni --help` for auto-generated, always-in-sync docs)
- `__main__.py` — enables `python -m omni`

Point at a non-default host with `--llm-host http://some-host:11434`
or the `LLM_HOST` env var (checked in that order).

If your LLM endpoint sits behind an authenticated proxy, set the key via
environment variable rather than the CLI flag — it avoids the token landing
in your shell history:
```bash
export LLM_API_KEY="sk-..."
omni "task" --project-root ./repo --llm-host http://your-host:8080
```

If you keep seeing "Thinking… (retry N/max_retries)" in the terminal, that's
usually a client-side request timeout, not the server being down — a large
local model can easily take longer than the default to respond to an
agentic tool-calling turn. Raise it with `--llm-timeout <seconds>` (default
`300`; applies to chat, intent parsing, and history compaction alike).
