"""Thin async MCP client: spawns the built-in tool server as a subprocess
scoped to a project root, optionally alongside any number of additional
("custom") MCP servers a user configures — local (stdio subprocess) or
remote (SSE / Streamable HTTP) — so the agent's tool set isn't limited to
what ships in this package, and new tools can be added without touching
this codebase at all.

Tool schemas from every connected server are merged into one list for the
model. Built-in tools keep their plain names (`read_file`, `write_file`,
...); tools from a custom server are namespaced as `<server_name>__<tool>`
so they can't collide with the built-ins or with each other.
"""

import asyncio
import base64
import json
import os
import re
import shlex
import sys
import time
from contextlib import AsyncExitStack

from anyio import BrokenResourceError
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .llm_client import LLMError, embed

try:
    BaseExceptionGroup  # builtin on Python 3.11+
except NameError:
    from exceptiongroup import BaseExceptionGroup  # backport anyio itself depends on pre-3.11


def _is_benign_shutdown_race(exc: BaseException) -> bool:
    """anyio's stdio_client runs its own stdout-reader task in a task group
    alongside the ClientSession built on top of it. AsyncExitStack closes
    contexts in reverse-entry order, so ClientSession's streams get closed
    BEFORE stdio_client's task group is torn down — if that reader task is
    mid-`send()` of a just-parsed message at that exact moment, the now-
    closed receive side raises BrokenResourceError. Harmless: everything is
    being shut down anyway. Recognizes it whether it surfaces directly or
    wrapped in an (possibly nested) ExceptionGroup from the task group."""
    if isinstance(exc, BrokenResourceError):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return all(_is_benign_shutdown_race(e) for e in exc.exceptions)
    return False

_BUILTIN = "_builtin"
_TRANSPORTS = ("sse", "streamable_http")
_MIN_SIMILARITY = 0.35   # cosine-similarity floor below which a semantic match is discarded as noise
_MAX_SEARCH_RESULTS = 5  # cap how many tools one search_tools call can reveal at once

_LOCAL_NOMIC_MODEL = "nomic-local"     # sentinel embedding_model value: on-device via `nomic[local]`
_NOMIC_MODEL_NAME = "nomic-embed-text-v1.5"  # concrete model the local backend downloads/runs


class EmbeddingUnavailable(Exception):
    """Raised by either embedding backend (local nomic or a remote
    OpenAI-compatible embedding model) when it can't produce vectors — missing dependency, model not
    pulled, network error. Caught by search_mcp_tools() to fall back to
    keyword matching rather than breaking the tool call."""


_LEGACY_MCP_CONFIG_NAME = "mcp.json"  # pre-0.5.x name, auto-migrated by default_mcp_config_path()

_ENV_REF_RE = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _expand_env_values(server_name: str, mapping: dict, kind: str) -> dict:
    """Resolve `$VAR` / `${VAR}` references in a server spec's `headers` or
    `env` values from the surrounding environment, so the settings file (or
    an `--add-mcp-server ,bearer=$VAR` registration) can reference a secret
    by name without storing it on disk.

    Raises ValueError naming the unset variable rather than letting
    os.path.expandvars silently pass an unexpanded "$VAR" through as the
    literal value — which would fail later as a confusing 401 or a server
    that misbehaves on a nonsense credential."""
    if not mapping:
        return mapping
    expanded = {}
    for key, value in mapping.items():
        if isinstance(value, str):
            missing = [
                m.group(1) or m.group(2)
                for m in _ENV_REF_RE.finditer(value)
                if (m.group(1) or m.group(2)) not in os.environ
            ]
            if missing:
                raise ValueError(
                    f"MCP server {server_name!r} {kind} {key!r} references unset environment "
                    f"variable(s): {', '.join(sorted(set(missing)))}"
                )
            value = os.path.expandvars(value)
        expanded[key] = value
    return expanded


def default_mcp_config_path() -> str:
    """Global, cross-session settings file — ~/.omni-coder/omni-coder-settings.json.
    MCP servers registered here (via `--add-mcp-server`) live under its
    `mcpServers` key and are available on every future run automatically,
    without passing --mcp-config/--mcp-server. Other top-level keys are left
    alone by save_mcp_config, so this file can hold non-MCP settings too.

    Migrates a pre-existing `mcp.json` (this file's old name) to the new
    name on first use, so servers registered before the rename keep working
    with no user action."""
    directory = os.path.join(os.path.expanduser("~"), ".omni-coder")
    path = os.path.join(directory, "omni-coder-settings.json")
    legacy = os.path.join(directory, _LEGACY_MCP_CONFIG_NAME)
    if not os.path.exists(path) and os.path.exists(legacy):
        try:
            os.rename(legacy, path)
        except OSError:
            return legacy  # couldn't migrate (permissions, etc.) — keep using the old file
    return path


def save_mcp_config(path: str, servers: dict) -> None:
    """Write `servers` to the file's `mcpServers` key, preserving every other
    top-level key already in it — this is a general settings file, so
    registering an MCP server must not clobber unrelated settings."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}  # unreadable/corrupt — rewrite rather than refusing to register
    data["mcpServers"] = servers
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _split_command(s: str) -> list:
    """Split a command string into tokens, Windows-path-safe. shlex.split()'s
    default POSIX mode treats backslash as an escape character, which
    silently eats every backslash in a Windows path (`C:\\Users\\...` becomes
    `C:Users...`) — so split in non-POSIX mode instead (preserves backslashes
    literally) and manually strip matching quotes non-POSIX mode leaves on
    quoted tokens."""
    tokens = shlex.split(s, posix=False)
    cleaned = []
    for t in tokens:
        if len(t) >= 2 and t[0] == t[-1] and t[0] in ('"', "'"):
            t = t[1:-1]
        cleaned.append(t)
    return cleaned


def parse_mcp_server_specs(specs: list) -> dict:
    """Parse repeatable `--mcp-server` CLI values into the same shape
    load_mcp_config() returns, so both sources can be merged uniformly.

    Two forms:
      - stdio (local subprocess):  "name=command arg1 arg2 ..."
      - remote (SSE / Streamable HTTP): "name=http://host/mcp/sse" or
        "name=http://host/mcp,streamable_http" (comma + transport to pick
        Streamable HTTP instead of the default SSE)

    Either form may end with a trailing ",defer" to mark the server for
    deferred tool loading (see MCPToolClient) — e.g.
    "docs=node docs-server.js,defer" or "weather=http://host/mcp,streamable_http,defer".

    A remote (URL) form may also carry ",bearer=<token>", which becomes an
    `Authorization: Bearer <token>` header on every request to that server.
    The value may be an environment-variable reference ("$TOKEN" or
    "${TOKEN}"), resolved at connect time rather than here — so
    `--add-mcp-server` persists only the reference, keeping the real secret
    out of both the settings file and your shell history. ",defer" and
    ",bearer=" may appear in either order.
    """
    servers = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f'Invalid --mcp-server value {spec!r} — expected "name=command args..."')
        name, rest = spec.split("=", 1)
        name, rest = name.strip(), rest.strip()
        if not name:
            raise ValueError(f'Invalid --mcp-server value {spec!r} — expected "name=command args..."')

        # Strip the trailing ",defer" / ",bearer=..." suffixes in whichever
        # order they were given, leaving `rest` as just the command or URL.
        defer, bearer = False, None
        while True:
            if rest.lower().endswith(",defer"):
                rest = rest[: -len(",defer")].strip()
                defer = True
                continue
            marker = rest.rfind(",bearer=")
            if marker != -1 and "," not in rest[marker + len(",bearer="):]:
                bearer = rest[marker + len(",bearer="):].strip()
                rest = rest[:marker].strip()
                if not bearer:
                    raise ValueError(f'Empty ",bearer=" token for MCP server {name!r}')
                continue
            break

        if rest.startswith("http://") or rest.startswith("https://"):
            if "," in rest:
                url, transport = (p.strip() for p in rest.rsplit(",", 1))
            else:
                url, transport = rest, "sse"
            if transport not in _TRANSPORTS:
                raise ValueError(f"Invalid --mcp-server transport {transport!r} for {name!r} — expected one of {_TRANSPORTS}")
            servers[name] = {"url": url, "transport": transport, "defer": defer}
            if bearer:
                servers[name]["headers"] = {"Authorization": f"Bearer {bearer}"}
        else:
            if bearer:
                raise ValueError(
                    f'",bearer=" is only valid for remote (http/https) MCP servers — {name!r} is a local command'
                )
            parts = _split_command(rest)
            if not parts:
                raise ValueError(f'Invalid --mcp-server value {spec!r} — expected "name=command args..."')
            servers[name] = {"command": parts[0], "args": parts[1:], "defer": defer}
    return servers


def load_mcp_config(path: str) -> dict:
    """Read a Claude-Desktop-style MCP config file:

        {
          "mcpServers": {
            "local-server": {
              "command": "python",
              "args": ["-m", "my_tools_server"],
              "env": {"SOME_VAR": "value"}
            },
            "remote-server": {
              "url": "https://example.com/mcp/sse",
              "transport": "sse",
              "headers": {"Authorization": "Bearer ..."}
            }
          }
        }

    Either entry may add `"defer": true` to keep that server's tools out of
    the model's default tool list (see MCPToolClient.list_llm_tools) — the
    model discovers them on demand via a synthesized search_tools tool.

    Returns the `mcpServers` mapping, or raises ValueError if the file can't
    be read/parsed, or a server entry has neither `command` (stdio) nor
    `url` (sse/streamable_http), or specifies an unknown `transport`.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not read MCP config {path!r}: {e}") from e

    servers = data.get("mcpServers", {})
    for name, spec in servers.items():
        if "url" in spec:
            transport = spec.get("transport", "sse")
            if transport not in _TRANSPORTS:
                raise ValueError(f"MCP server {name!r} in {path!r} has unknown transport {transport!r} — expected one of {_TRANSPORTS}")
        elif "command" not in spec:
            raise ValueError(f"MCP server {name!r} in {path!r} must have either 'command' (stdio) or 'url' (sse/streamable_http)")
    return servers


def _mcp_schema_to_tool_schema(tool, exposed_name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": exposed_name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


_SEARCH_TOOLS_NAME = "search_tools"


def _search_tools_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": _SEARCH_TOOLS_NAME,
            "description": (
                "Search for additional tools from MCP servers registered with deferred "
                "loading — their schemas aren't in your tool list yet to save context. "
                "Call this with keywords describing the capability you need (e.g. a tool "
                "name, or what you're trying to do); matching tools are then loaded and "
                "become callable on your next turn. Leave query empty to load everything "
                "remaining."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to match against deferred tool names/descriptions, e.g. \"weather forecast\".",
                    },
                },
                "required": ["query"],
            },
        },
    }


_LIST_RESOURCES_NAME = "list_resources"
_READ_RESOURCE_NAME = "read_resource"


def _resource_tool_schemas(resources: dict) -> list:
    """Model-facing tools for the MCP "Resources" capability — readable
    context a server publishes by URI. Only offered when a connected server
    actually publishes at least one resource (see list_llm_tools), so a
    setup with no resource-providing servers pays nothing for them.

    A short catalog of the known URIs goes straight into read_resource's
    description, so the common case (a handful of resources) needs no
    list_resources round trip at all."""
    readable = [uri for uri, info in resources.items() if not info.get("template")]
    catalog = ", ".join(readable[:20])
    if len(readable) > 20:
        catalog += f", … ({len(readable) - 20} more — call {_LIST_RESOURCES_NAME})"
    return [
        {
            "type": "function",
            "function": {
                "name": _LIST_RESOURCES_NAME,
                "description": (
                    "List readable resources published by the connected MCP servers — "
                    "files, docs, or records exposed as context, addressed by URI. Use "
                    "this to discover what's available, then read one with read_resource. "
                    "Takes no arguments."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": _READ_RESOURCE_NAME,
                "description": (
                    "Read the contents of an MCP resource by its URI. This reads context "
                    "published by an MCP server — unrelated to read_file, which reads a "
                    f"path in the project directory. Currently available: {catalog or '(none)'}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uri": {
                            "type": "string",
                            "description": 'Resource URI exactly as listed, e.g. "file:///readme.md".',
                        },
                    },
                    "required": ["uri"],
                },
            },
        },
    ]


class MCPToolClient:
    """Use as an async context manager, one instance per agent run:

        async with MCPToolClient(project_root, mcp_config_path="mcp.json",
                                  extra_servers={"toy": {"command": "python", "args": [...]}}) as client:
            schemas = await client.list_llm_tools()
            result = await client.call_tool("edit_file", {...})

    `mcp_config_path` and `extra_servers` can be used together — servers
    from both are started; on a name collision, `extra_servers` wins.
    """

    def __init__(self, project_root: str, server_path: str = None,
                 mcp_config_path: str = None, extra_servers: dict = None,
                 embedding_model: str = "", llm_host: str = None, llm_api_key: str = None,
                 mcp_log_path: str = "mcp_servers.log"):
        self.project_root = project_root
        # stderr from every stdio-transport server (built-in + custom) is
        # redirected here instead of the terminal — opened lazily in
        # __aenter__ and closed via self._stack on exit.
        self.mcp_log_path = mcp_log_path or "mcp_servers.log"
        self._server_log_file = None
        # Default: run the built-in server as `python -m <package>.mcp_server`
        # rather than by file path — mcp_server.py uses relative imports
        # (it's part of this package), which only resolve when it's launched
        # as a module, not executed as a standalone script. `server_path` is
        # an escape hatch for pointing the *built-in* slot at a different
        # server entirely.
        self.server_args = [server_path] if server_path else ["-m", f"{__package__}.mcp_server"]
        self.mcp_config_path = mcp_config_path
        self.extra_servers = extra_servers or {}
        # Embedding backend for semantic search_tools ranking:
        # "nomic-local" (default) -> on-device via the `nomic` package;
        # any other name -> a remote OpenAI-compatible embedding model;
        # "" -> disabled (falls back to keyword matching in search_mcp_tools).
        self.embedding_model = embedding_model or ""
        self.llm_host = llm_host
        self.llm_api_key = llm_api_key
        self._stack = AsyncExitStack()
        self._sessions: dict = {}     # server name -> ClientSession (only successfully-connected ones)
        self._tool_owner: dict = {}   # exposed tool name -> (server name, real tool name)
        self._prompt_owner: dict = {}  # exposed prompt name ("server:prompt") -> (server name, real prompt name)
        self._resource_owner: dict = {}  # resource uri -> server name that published it (see list_resources)
        self._resources: dict = {}       # cached list_resources() catalog, filled on connect
        self._deferred_servers: set = set()  # server names registered with "defer": true
        self._deferred_tools: dict = {}       # exposed name -> schema, still hidden from the model
        self._revealed: set = set()           # exposed names of deferred tools search_tools has surfaced
        self._tool_embeddings: dict = {}      # exposed name -> embedding vector, cached across searches
        self._server_specs: dict = {}   # server name -> spec, EVERY configured server (for /mcp, even ones that failed)
        self._connected_at: dict = {}   # server name -> time.monotonic() at successful connect
        self._connect_errors: dict = {}  # server name -> error message, for servers that failed to connect

    async def _connect(self, name: str, spec: dict) -> ClientSession:
        try:
            if "url" in spec:
                transport = spec.get("transport", "sse")
                headers = _expand_env_values(name, spec.get("headers"), "header")
                if transport == "streamable_http":
                    read, write, _ = await self._stack.enter_async_context(
                        streamablehttp_client(spec["url"], headers=headers)
                    )
                else:
                    read, write = await self._stack.enter_async_context(
                        sse_client(spec["url"], headers=headers)
                    )
            else:
                params = StdioServerParameters(
                    command=spec["command"], args=spec.get("args", []),
                    env={**os.environ, **(_expand_env_values(name, spec.get("env"), "env var") or {})},
                )
                self._server_log_file.write(
                    f"\n=== [{name}] {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"starting: {spec['command']} {' '.join(spec.get('args', []))} ===\n"
                )
                self._server_log_file.flush()
                read, write = await self._stack.enter_async_context(
                    stdio_client(params, errlog=self._server_log_file)
                )
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return session
        except Exception as e:
            desc = spec.get("url") or f"{spec.get('command')} {' '.join(spec.get('args', []))}"
            raise RuntimeError(f"Failed to start MCP server {name!r} ({desc}): {e}") from e

    async def __aenter__(self):
        # Opened once per client and reused across every stdio server's
        # errlog= — closed automatically via self._stack on __aexit__.
        self._server_log_file = self._stack.enter_context(open(self.mcp_log_path, "a", encoding="utf-8"))

        builtin_spec = {
            "command": sys.executable, "args": self.server_args,
            "env": {"AGENT_PROJECT_ROOT": self.project_root},
        }
        # The built-in server provides the core file/shell tools — a failure
        # there is fatal, so it still raises. Custom servers are optional
        # add-ons: one failing shouldn't take down the whole session, so
        # each is caught individually and recorded for server_status()/the
        # /mcp REPL command to report, instead of aborting startup.
        self._server_specs[_BUILTIN] = builtin_spec
        self._sessions[_BUILTIN] = await self._connect(_BUILTIN, builtin_spec)
        self._connected_at[_BUILTIN] = time.monotonic()

        servers = dict(load_mcp_config(self.mcp_config_path)) if self.mcp_config_path else {}
        servers.update(self.extra_servers)  # CLI-specified --mcp-server entries win on name clash
        for name, spec in servers.items():
            self._server_specs[name] = spec
            try:
                self._sessions[name] = await self._connect(name, spec)
                self._connected_at[name] = time.monotonic()
                if spec.get("defer"):
                    self._deferred_servers.add(name)
            except RuntimeError as e:
                self._connect_errors[name] = str(e)

        # Catalog resources once here rather than on every list_llm_tools()
        # call (which runs per turn) — it decides whether the resource tools
        # are offered to the model at all, and populates read_resource()'s
        # uri -> server routing. The list_resources tool re-reads it live, so
        # a server publishing resources later is still reachable that way.
        try:
            self._resources = await self.list_resources()
        except Exception:
            self._resources = {}
        return self

    async def __aexit__(self, *exc_info):
        try:
            await self._stack.aclose()
        except BaseException as e:
            if not _is_benign_shutdown_race(e):
                raise

    def server_status(self) -> list:
        """One entry per configured server (built-in + every custom one),
        regardless of whether it actually connected, for the /mcp REPL
        command. Built-in first, then custom servers in configured order.
        Each entry: {name, connected, connected_for (seconds, None if not
        connected), error (None if connected), deferred, tool_count, target}."""
        now = time.monotonic()
        entries = []
        for name, spec in self._server_specs.items():
            connected = name in self._sessions
            entries.append({
                "name": "built-in" if name == _BUILTIN else name,
                "connected": connected,
                "connected_for": (now - self._connected_at[name]) if connected else None,
                "error": self._connect_errors.get(name),
                "deferred": name in self._deferred_servers,
                "tool_count": sum(1 for owner in self._tool_owner.values() if owner[0] == name),
                "target": spec.get("url") or f"{spec.get('command', '')} {' '.join(spec.get('args', []))}".strip(),
            })
        return entries

    async def list_llm_tools(self) -> list:
        """Schemas for tools the LLM is allowed to call, merged across every
        connected server. Internal underscore-prefixed built-in tools
        (previews, existence checks) are held back — the agent still calls
        those directly for its own logic. Tools from custom servers are
        namespaced as `<server_name>__<tool_name>`.

        Tools belonging to a server registered with "defer": true are held
        back too (not counted against the model's context) until
        search_tools() reveals them by name — mirroring how the harness's
        own ToolSearch keeps rarely-used tools out of the default set. Call
        this again after a search_tools call to pick up newly revealed
        schemas; it's safe to call repeatedly."""
        schemas = []
        self._tool_owner = {}
        self._deferred_tools = {}
        for server_name, session in self._sessions.items():
            result = await session.list_tools()
            for tool in result.tools:
                if server_name == _BUILTIN:
                    if tool.name.startswith("_"):
                        continue
                    exposed_name = tool.name
                else:
                    exposed_name = f"{server_name}__{tool.name}"
                self._tool_owner[exposed_name] = (server_name, tool.name)
                schema = _mcp_schema_to_tool_schema(tool, exposed_name)
                if server_name in self._deferred_servers and exposed_name not in self._revealed:
                    self._deferred_tools[exposed_name] = schema
                else:
                    schemas.append(schema)
        if self._deferred_tools:
            schemas.append(_search_tools_schema())
        if self._resources:
            schemas.extend(_resource_tool_schemas(self._resources))
        return schemas

    @staticmethod
    def _tool_embedding_text(exposed_name: str, schema: dict) -> str:
        return f"{exposed_name}: {schema['function'].get('description', '')}"

    def _keyword_match(self, query: str) -> list:
        """Fallback used when semantic ranking is disabled (no embedding_model
        configured) or the embedding call fails: every word in `query` must
        appear (case-insensitive substring) in the tool's name/description."""
        terms = query.lower().split()
        return [
            exposed_name for exposed_name, schema in self._deferred_tools.items()
            if not terms or all(t in self._tool_embedding_text(exposed_name, schema).lower() for t in terms)
        ]

    async def _embed_local_nomic(self, texts: list, task_type: str) -> list:
        """On-device embeddings via the `nomic` package's local (GPT4All)
        backend — no server, no API key. Needs `pip install "nomic[local]"`;
        the model itself downloads on first use. `nomic.embed.text` is
        synchronous/blocking, so it's run off the event loop in a thread."""
        try:
            from nomic import embed as nomic_embed
        except ImportError as e:
            raise EmbeddingUnavailable(
                'the "nomic" package isn\'t installed — run `pip install "nomic[local]"`, '
                "or set --embedding-model to a remote OpenAI-compatible embedding model instead"
            ) from e

        def _run():
            output = nomic_embed.text(
                texts=texts, model=_NOMIC_MODEL_NAME, task_type=task_type, inference_mode="local",
            )
            return output["embeddings"]

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            raise EmbeddingUnavailable(f"local nomic embedding failed: {e}") from e

    async def _embed_texts(self, texts: list, task_type: str) -> list:
        """Route to the local nomic[local] backend (the default,
        self.embedding_model == _LOCAL_NOMIC_MODEL) or a remote OpenAI-compatible
        embedding model (any other non-empty self.embedding_model)."""
        if self.embedding_model == _LOCAL_NOMIC_MODEL:
            return await self._embed_local_nomic(texts, task_type)
        try:
            return await embed(self.embedding_model, texts, base_url=self.llm_host, api_key=self.llm_api_key)
        except LLMError as e:
            raise EmbeddingUnavailable(str(e)) from e

    async def _semantic_match(self, query: str) -> list:
        """Rank deferred tools by cosine similarity between the query and
        each tool's name+description, embedded via self.embedding_model.
        Embeddings are cached per exposed name across calls since tool
        descriptions don't change within a client's lifetime. Returns the
        top matches above _MIN_SIMILARITY, best first."""
        to_embed = [name for name in self._deferred_tools if name not in self._tool_embeddings]
        if to_embed:
            texts = [self._tool_embedding_text(name, self._deferred_tools[name]) for name in to_embed]
            vectors = await self._embed_texts(texts, task_type="search_document")
            for name, vector in zip(to_embed, vectors):
                self._tool_embeddings[name] = vector

        (query_vector,) = await self._embed_texts([query], task_type="search_query")

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a, norm_b = sum(x * x for x in a) ** 0.5, sum(y * y for y in b) ** 0.5
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

        scored = sorted(
            ((name, cosine(query_vector, self._tool_embeddings[name])) for name in self._deferred_tools),
            key=lambda pair: pair[1], reverse=True,
        )
        return [name for name, score in scored[:_MAX_SEARCH_RESULTS] if score >= _MIN_SIMILARITY]

    async def search_mcp_tools(self, query: str) -> str:
        """Reveal deferred tools matching `query` — semantically (cosine
        similarity over embeddings, computed either on-device via `nomic`
        or by a remote OpenAI-compatible model, per self.embedding_model) when
        enabled, falling back to keyword substring matching otherwise or if
        the embedding backend errors. Empty query reveals everything still
        hidden. Matches are recorded in `self._revealed` so the next
        list_llm_tools() call includes their schemas. Returns a
        human-readable summary for the model."""
        query = (query or "").strip()
        if not self._deferred_tools:
            return "No deferred tools remain — everything is already loaded."

        fallback_note = ""
        if not query:
            matches = list(self._deferred_tools)
        elif self.embedding_model:
            try:
                matches = await self._semantic_match(query)
            except EmbeddingUnavailable as e:
                matches = self._keyword_match(query)
                fallback_note = f"(semantic search failed ({e}), fell back to keyword match)\n"
        else:
            matches = self._keyword_match(query)

        if not matches:
            remaining = sorted(self._deferred_tools)
            return f"{fallback_note}No deferred tools matched {query!r}. Still hidden: {', '.join(remaining)}"
        self._revealed.update(matches)
        lines = [f"{fallback_note}Loaded {len(matches)} tool(s) — now available to call:"]
        for exposed_name in matches:
            desc = self._deferred_tools[exposed_name]["function"].get("description", "").strip().splitlines()[:1]
            lines.append(f"- {exposed_name}: {desc[0] if desc else ''}")
        return "\n".join(lines)

    async def list_prompts(self) -> dict:
        """Prompt templates exposed by every connected MCP server — the MCP
        "Prompts" capability, distinct from tools: user-invocable templates
        rather than model-callable functions. Namespaced as
        "<server_name>:<prompt_name>" (a colon, unlike tools' "__", so the
        REPL can tell a `/server:prompt` command from a plain tool name).
        Servers that don't implement prompts/list (the built-in tool server
        doesn't declare any) are skipped rather than raising.

        Returns {exposed_name: {"description": str, "arguments": [{"name", "description", "required"}, ...]}}.
        """
        prompts = {}
        self._prompt_owner = {}
        for server_name, session in self._sessions.items():
            try:
                result = await session.list_prompts()
            except Exception:
                continue
            for p in result.prompts:
                exposed_name = f"{server_name}:{p.name}"
                self._prompt_owner[exposed_name] = (server_name, p.name)
                prompts[exposed_name] = {
                    "description": p.description or "",
                    "arguments": [
                        {"name": a.name, "description": a.description or "", "required": bool(a.required)}
                        for a in (p.arguments or [])
                    ],
                }
        return prompts

    async def get_prompt(self, exposed_name: str, arguments: dict = None) -> str:
        """Fetch a prompt template (by its "server:prompt" exposed name, from
        list_prompts()) and flatten its rendered messages into plain text,
        ready to feed to the model as if the user had typed it."""
        if exposed_name not in self._prompt_owner:
            raise ValueError(f"Unknown prompt {exposed_name!r} — run /prompts or check server name/spelling.")
        server_name, real_name = self._prompt_owner[exposed_name]
        session = self._sessions[server_name]
        result = await session.get_prompt(real_name, arguments or None)
        parts = []
        for m in result.messages:
            text = m.content.text if hasattr(m.content, "text") else str(m.content)
            parts.append(text)
        return "\n\n".join(parts)

    async def list_resources(self, include_templates: bool = False) -> dict:
        """Resources exposed by every connected MCP server — the MCP
        "Resources" capability: readable context (files, docs, records) a
        server publishes by URI, distinct from tools (callable) and prompts
        (user-invocable templates). Servers that don't implement
        resources/list — including the built-in tool server — are skipped
        rather than raising.

        Keyed by the resource's own URI, since that's how MCP identifies one
        and how read_resource() takes it. If two servers publish the same
        URI the first connected wins and the collision is recorded in that
        entry's "shadowed_by" — read_resource(uri, server=...) can still
        reach the other one explicitly.

        With `include_templates=True`, also returns parameterized resource
        templates (URI patterns like "file:///logs/{date}.log"), marked
        `"template": True`. Those can't be passed to read_resource() as-is —
        the caller has to fill in the placeholders first.

        Returns {uri: {"server", "name", "description", "mime_type", "size",
        "template", "shadowed_by"}}.
        """
        resources = {}
        self._resource_owner = {}
        for server_name, session in self._sessions.items():
            listings = [("list_resources", "resources", False)]
            if include_templates:
                listings.append(("list_resource_templates", "resourceTemplates", True))
            for method, attr, is_template in listings:
                try:
                    result = await getattr(session, method)()
                except Exception:
                    continue
                for r in getattr(result, attr, []) or []:
                    # Templates carry uriTemplate; concrete resources carry uri.
                    uri = str(getattr(r, "uriTemplate", None) or getattr(r, "uri", ""))
                    if not uri:
                        continue
                    if uri in resources:
                        resources[uri].setdefault("shadowed_by", []).append(server_name)
                        continue
                    if not is_template:
                        self._resource_owner[uri] = server_name
                    resources[uri] = {
                        "server": server_name,
                        "name": getattr(r, "name", "") or "",
                        "description": getattr(r, "description", "") or "",
                        "mime_type": getattr(r, "mimeType", None) or "",
                        "size": getattr(r, "size", None),
                        "template": is_template,
                        "shadowed_by": [],
                    }
        return resources

    async def read_resource(self, uri: str, server: str = None) -> str:
        """Read a resource by URI and flatten its contents to text, ready to
        feed to the model. `server` forces which connected server to ask,
        for the rare case where two publish the same URI; otherwise the URI
        is routed to whichever server listed it (call list_resources() first
        to populate that routing table).

        Binary (blob) contents aren't base64-dumped into the returned text —
        that would flood the model's context with unusable bytes — they're
        replaced by a short marker naming the mime type and decoded size.
        """
        if server is not None:
            if server not in self._sessions:
                raise ValueError(f"Unknown MCP server {server!r} — connected: {sorted(self._sessions)}")
            server_name = server
        elif uri in self._resource_owner:
            server_name = self._resource_owner[uri]
        else:
            raise ValueError(
                f"Unknown resource {uri!r} — call list_resources() first, or pass server= explicitly."
            )

        result = await self._sessions[server_name].read_resource(uri)
        parts = []
        for content in result.contents:
            if hasattr(content, "text"):
                parts.append(content.text)
            elif hasattr(content, "blob"):
                mime = getattr(content, "mimeType", None) or "application/octet-stream"
                try:
                    size = len(base64.b64decode(content.blob))
                    parts.append(f"[binary {mime}, {size} bytes — not shown]")
                except (ValueError, TypeError):
                    parts.append(f"[binary {mime} — not shown]")
        return "\n\n".join(parts)

    async def call_tool(self, name: str, args: dict) -> str:
        if name == _SEARCH_TOOLS_NAME:
            return await self.search_mcp_tools(args.get("query", ""))
        if name == _LIST_RESOURCES_NAME:
            self._resources = await self.list_resources()  # refresh; servers may publish more over time
            if not self._resources:
                return "(no resources published by the connected MCP servers)"
            lines = []
            for uri, info in self._resources.items():
                detail = " — ".join(p for p in (info["name"], info["description"]) if p)
                mime = f" [{info['mime_type']}]" if info["mime_type"] else ""
                lines.append(f"{uri}{mime}{f' — {detail}' if detail else ''}")
            return "\n".join(lines)
        if name == _READ_RESOURCE_NAME:
            uri = args.get("uri", "")
            if not uri:
                return "ERROR: read_resource requires a 'uri' argument."
            try:
                return await self.read_resource(uri)
            except ValueError as e:
                return f"ERROR: {e}"
            except Exception as e:
                return f"ERROR: reading resource {uri!r} failed: {e}"
        if name in self._deferred_tools:
            return f"ERROR: tool {name!r} isn't loaded yet — call search_tools with a matching query first."
        # Internal built-in tools (_preview_edit, etc.) are never registered
        # in _tool_owner since list_llm_tools() filters them out — the
        # (server, name) default below routes them straight to the built-in
        # session under their real name.
        server_name, real_name = self._tool_owner.get(name, (_BUILTIN, name))
        session = self._sessions[server_name]
        result = await session.call_tool(real_name, args)
        text = "".join(c.text for c in result.content if hasattr(c, "text"))
        return f"ERROR: {text}" if result.isError else text

    async def preview_edit(self, path: str, old_str: str, new_str: str):
        raw = await self.call_tool("_preview_edit", {"path": path, "old_str": old_str, "new_str": new_str})
        ok = raw.startswith("OK\n")
        msg = raw.split("\n", 1)[1] if "\n" in raw else raw
        return ok, msg

    async def preview_write(self, path: str, content: str, overwrite: bool = False):
        raw = await self.call_tool("_preview_write", {"path": path, "content": content, "overwrite": overwrite})
        is_new = raw.startswith("NEW\n")
        preview = raw.split("\n", 1)[1] if "\n" in raw else raw
        return is_new, preview

    async def file_exists(self, path: str) -> bool:
        raw = await self.call_tool("_file_exists", {"path": path})
        return raw.strip() == "true"
