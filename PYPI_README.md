# 🐙 Omni Coder

[![tests](https://img.shields.io/badge/tests-758%20passed-brightgreen)](#-tests)
[![coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)](#-tests)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

An AI coding agent that plans, edits, and tests code by driving Qwen Coder
(or any OpenAI-compatible model) through a scoped set of file and shell
tools, with human approval on every write, edit, or shell command.

## 📦 Install

```bash
pip install omni-coder
ollama pull qwen3-coder:30b   # example: pulling the default model via Ollama
```

## ▶️ Usage

```bash
omni "Add type hints to utils.py, then run the test suite" \
    --project-root ./myrepo
```

Equivalent: `python -m omni "..." --project-root ./myrepo`.

Omit the task string to enter an interactive session instead:

```bash
omni --project-root ./myrepo
```

Run `omni --help` for the full option list.

## ✨ Features

- **Structured intent parsing** — the raw task is classified (bug fix,
  feature, refactor, risk level, target files) before any action is taken,
  and high-risk tasks force human approval even under `--auto-approve`.
- **Session persistence** — every message is saved to SQLite as the run
  happens. Resume a previous run by id or a name you gave it
  (`--resume`), browse saved sessions (`--list-sessions`), or delete one
  (`--delete-session`).
- **Interactive mode** — drop into a REPL that keeps the model connection
  and tool session alive across turns. Ctrl-C during a running turn cancels
  just that turn instead of killing the session — you land back at the
  prompt and can keep going. Type `/` for a completion menu of every
  command, including `/model` (switch models) and `/server:prompt` (run an
  MCP prompt template).
- **Automatic context compaction** — once the running conversation exceeds
  `--context-char-budget` (default 200k chars), older messages are replaced
  with an LLM-written summary instead of growing forever or being silently
  dropped. Trigger it manually anytime with `/compact`.
- **Human-in-the-loop approval** — every write, edit, or shell command
  shows a diff or command preview before you confirm (diffs render with
  line numbers and red/green highlighting), unless explicitly marked safe
  or run with `--auto-approve`.
- **Retry and recovery** — transient model failures retry with backoff;
  malformed tool-call output is caught and reported back to the model
  instead of crashing the run.
- **Codebase exploration tools** — regex content search with glob
  filtering, pattern-based file discovery, directory listing, and a full
  git toolset (status/log/diff/show/branch/fetch read-only; add/commit/
  pull/push approval-gated), all skipping noise directories (`.git`,
  `node_modules`, build output).
- **Persistent project memory** — the agent can save durable notes (a
  `save_memory` tool call) to a per-project `agent_memory.md`, auto-loaded
  into the system prompt at the start of every new session.
- **Extensible via custom MCP servers** — point at any MCP server, local
  (stdio) or remote (SSE / Streamable HTTP), and its tools merge into the
  model's toolset automatically, no code changes required. Register one
  permanently (`--add-mcp-server`, available on every future run) or add
  one per run (`--mcp-server`/`--mcp-config`). One server failing to
  connect doesn't take down the session — check `/mcp` for live ✅/❌
  status per server, and `--mcp-log-path` for their stderr output.
- **Inspect a server's tools** — `/mcp tools <name>` lists what one server
  exposes (the name the model calls each by, plus its description), flagging
  tools that are `deferred`, `revealed`, or `internal`.
- **Hot-restart a server you're editing** — `/mcp restart <name>` (or
  `all`) reconnects just that server without leaving the REPL, picking up
  code *and* config changes and re-listing its tools/prompts/resources. Also
  how you retry a server that failed to connect, once you've fixed it.
- **Deferred tool loading + semantic search_tools** — register a custom MCP
  server with `--defer` and its tools stay out of the model's context until
  a synthesized `search_tools` tool loads matching ones on demand, ranked by
  on-device embeddings (`nomic-local`, default) or a remote OpenAI-compatible
  embedding model, with automatic keyword-match fallback.
- **MCP resources** — readable context a server publishes by URI (coding
  standards, API schemas, records). Browse them with `/resources` and read
  one with `/resources <uri>`; the model gets matching read-only
  `list_resources`/`read_resource` tools automatically whenever a connected
  server publishes any, so you can just say "read the coding standards
  resource, then fix utils.py to match".

## 🏗️ Architecture

Tools are served over the Model Context Protocol (MCP), not called
in-process — the agent is an MCP *client* that talks to a tool server over
stdio:

```
+-----------------------------+
|          CLI / REPL         |
+-----------------------------+
               |
               v
+-----------------------------+
|          Agent loop         |
|  parse intent, call model,  |
|  approve, execute, persist  |
+-----------------------------+
               |
               v
+-----------------------------+
|          MCP client         |
|  built-in + custom servers  |
|  merged into one tool list. |
| "defer"-registered servers  |
| hold tools back for on-     |
| demand search_tools lookup  |
+-----------------------------+
               |
 stdio / SSE / streamable-http
               v
+-----------------------------+
|        MCP server(s)        |
+-----------------------------+
               |
               v
+-----------------------------+
|            Tools            |
|    read / write / edit /    |
|        search / shell       |
+-----------------------------+
```

Because tools are exposed over MCP, any MCP-compatible client — Claude
Desktop, another agent framework, a different model entirely — can reach
the exact same toolset, approval-preview logic, and path scoping. The
reverse also holds: any additional MCP server — local (stdio) or remote
(SSE / Streamable HTTP) — can be plugged into this agent, and its tools
merge into the same list the model already sees —
```bash
omni --add-mcp-server "weather=python -m weather_mcp_server"     # local, stdio
omni --add-mcp-server "weather=https://example.com/mcp/sse"      # remote, SSE
omni "what's the forecast?"   # picked up automatically, every run from here on
```
Registrations are saved under the `mcpServers` key of
`~/.omni-coder/omni-coder-settings.json`, leaving any other key in that file
untouched.

A value after `name=` starting with `http://`/`https://` is treated as a
remote server (SSE by default, append `,streamable_http` for that transport
instead); anything else is a local command spawned over stdio — it doesn't
need to be `-m`-invokable, a standalone script's absolute path works too
(e.g. `"myserver=python C:/absolute/path/to/mcp_server.py"`).

**Authenticated remote servers** take a bearer token via a `,bearer=<token>`
suffix, which becomes an `Authorization: Bearer` header:
```bash
export DOCS_TOKEN="sk-..."
omni --add-mcp-server 'docs=https://example.com/mcp/sse,bearer=$DOCS_TOKEN'
```
Prefer that `$VAR` form over a literal token — `headers` and `env` values
(both in the settings file and in `--mcp-config` JSON) are resolved from the
environment at connect time, so only the variable *name* is written to disk,
never the secret. Use **single quotes** so your shell doesn't expand the
variable before this agent sees it. An unset variable is reported as a clear
error instead of being sent as a literal `$VAR`.

Append `,defer` (or pass `--defer` with `--add-mcp-server`) to keep a
server's tools out of the model's default tool list — it discovers them on
demand via `search_tools`, ranked semantically by default
(`pip install "omni-coder[local-embeddings]"` for on-device
embeddings, or point `--embedding-model` at a remote one instead;
`--embedding-model ""` falls back to plain keyword matching). See the full
README for details.

## ⚙️ Configuration

Point at any OpenAI-compatible host with `--llm-host` or the
`LLM_HOST` env var. If it sits behind an authenticated proxy, set
`LLM_API_KEY` as an environment variable rather than a CLI flag so the
key doesn't end up in shell history. Seeing repeated retries in the
terminal? That's usually a client-side timeout, not a dead server — raise
it with `--llm-timeout <seconds>` (default `300`).

## 🧪 Tests

758 tests, 94% branch coverage — hermetic (no model, server, or network
needed; every external boundary is mocked):
```bash
pip install -e ".[dev]"
pytest                 # whole suite
pytest --cov=omni      # with coverage
pytest -m "not live"   # skip the subprocess-spawning tests
```
See the full README for the per-module breakdown.

## 🔗 Links

Source, full documentation, and issue tracker:
https://github.com/HarryChen1995/omni-coder

## 📄 License

MIT
