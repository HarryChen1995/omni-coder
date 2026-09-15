"""Configuration for the coding agent."""

import json
import os
import re
from dataclasses import dataclass, field

# ---- the settings file -----------------------------------------------------
#
# ~/.omni-coder/omni-coder-settings.json, the same file MCP servers are
# registered in. It has always been a general settings file rather than an MCP
# one (save_mcp_config is careful to leave other top-level keys alone); these
# helpers are the non-MCP half of it, so a preference set once from the REPL
# survives into every later run.

THEME_COLOR_KEY = "themeColor"
SYSTEM_PROMPT_KEY = "systemPrompt"
CONTEXT_CHAR_BUDGET_KEY = "contextCharBudget"

_HEX_COLOR_RE = re.compile(r"#?[0-9a-fA-F]{6}")

# Below this a budget is self-defeating: the system prompt alone is ~900
# characters, so anything smaller compacts on every single step and spends a
# summarization call per turn to save nothing. It is a floor on the setting,
# not on the model's context.
MIN_CONTEXT_CHAR_BUDGET = 2_000


def normalize_hex_color(value: str) -> str:
    """`value` as "#rrggbb", or None if it isn't a 6-digit hex colour.

    The leading "#" is optional on the way in because both places a colour is
    typed — the --theme-color flag and /theme-color — are shells or shell-like
    prompts where "#" starts a comment often enough to be worth forgiving.
    Three-digit shorthand is deliberately not accepted: "#abc" is as likely to
    be a typo for a six-digit value as it is to be shorthand, and guessing
    wrong here silently recolours the whole UI."""
    text = (value or "").strip()
    if not _HEX_COLOR_RE.fullmatch(text):
        return None
    return text if text.startswith("#") else f"#{text}"


def settings_path() -> str:
    """Where the settings file lives. Imported lazily: mcp_client owns this
    path (it also migrates the file's old name), and it pulls in the MCP SDK,
    which nothing here otherwise needs."""
    from .mcp_client import default_mcp_config_path
    return default_mcp_config_path()


def load_settings(path: str = None) -> dict:
    """The settings file as a dict — empty if it's missing, unreadable or
    corrupt. A broken settings file must not stop the agent from starting; the
    cost of ignoring it is falling back to defaults, which is what a fresh
    install does anyway."""
    path = path or settings_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def save_setting(key: str, value, path: str = None) -> str:
    """Write one top-level key, preserving everything else in the file
    (`mcpServers` above all). `value` of None removes the key.

    Returns the path written to, so a caller can tell the user where the
    preference landed. Raises OSError if the file can't be written — a
    save that silently didn't is worse than an error message."""
    path = path or settings_path()
    data = load_settings(path)
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return path


# ---- the settings registry -------------------------------------------------
#
# One row per preference that can be set from the REPL and remembered across
# runs. Everything downstream is driven off this table rather than written
# per setting: the slash commands and their completion entries, the "set it,
# save it, reset it" handler, the /config listing, and the startup rule that a
# flag beats the saved value beats the built-in default. Adding a preference
# is adding a row.
#
# What is deliberately NOT here: --auto-approve (a saved "never ask me again"
# is a footgun that outlives the run that wanted it), --llm-api-key (a secret
# does not belong in a plaintext settings file — use $LLM_API_KEY), and the
# per-invocation paths (--project-root, --db-path, the log files), which
# describe one run rather than a preference.


def _text(value):
    """Any non-blank string. Blank is None so "/model   " reads as a
    question rather than as clearing the model to nothing."""
    text = str(value if value is not None else "").strip()
    return text or None


def _prose(value):
    """Text kept exactly as it was given, unless it is blank.

    Unlike _text this does not strip: a system prompt read from a file is
    the user's formatting, trailing newline included, and the REPL has
    already stripped what was typed at the prompt by the time it lands
    here."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _whole(minimum: int):
    """A whole number at or above `minimum`, or None.

    Digit-group separators are accepted ("200_000", "200,000"): the context
    budget is six digits long and the code that sets its default spells it
    200_000, and a number mistyped by a factor of ten is not obvious on
    screen. Anything under the minimum is refused rather than clamped —
    silently using a different number than the one typed is how a session
    ends up compacting every turn with no explanation."""
    def parse(value):
        if isinstance(value, bool):   # bool is an int; "true" is not a number
            return None
        if isinstance(value, int):
            number = value
        else:
            text = str(value if value is not None else "").strip().replace("_", "").replace(",", "")
            if not text.isdigit():
                return None
            number = int(text)
        return number if number >= minimum else None
    return parse


def _seconds(minimum: float):
    """A number of seconds at or above `minimum`. Floats are accepted because
    the timeouts are floats; the step caps use _whole instead."""
    def parse(value):
        if isinstance(value, bool):
            return None
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number if number >= minimum else None
    return parse


_TRUE = ("true", "yes", "on", "1")
_FALSE = ("false", "no", "off", "0")


def _flag(value):
    """A boolean, spelled however it was typed. None for anything else —
    an unrecognized word must not quietly read as False."""
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    return True if text in _TRUE else False if text in _FALSE else None


def _embedding_backend(value):
    """A model name, or "" for the keyword-matching fallback. The off
    switch needs a spelling here because the empty string can be passed to
    the flag but not typed at a slash command."""
    text = str(value if value is not None else "").strip()
    if text.lower() in ("off", "none", "disabled", "keyword"):
        return ""
    return text


@dataclass(frozen=True)
class Setting:
    """One saveable preference. `name` is both the slash command (hyphenated,
    no slash) and how it is listed; `field` is the AgentConfig field it sets;
    `key` is its top-level key in the settings file."""

    name: str
    field: str
    key: str
    parse: callable
    summary: str          # one line, for /config and the completion menu
    hint: str             # what a usable value looks like, for the error
    # When a change takes hold. "now" is the running loop; "session" is the
    # next new session (the system prompt is stored as a session's first
    # message when the session is created); "run" is the next start of omni
    # (the tool-side knobs are handed to the MCP server process as env when
    # it connects, and the connect timeout has already been spent by then).
    scope: str = "now"
    from_file: bool = False   # also settable as "<name> file <path>"
    # An environment variable that supplies the value when neither the flag
    # nor the settings file does. It sits below the saved preference on
    # purpose: a variable named DEFAULT_* is what to use when nothing else
    # says, not an override of what you deliberately saved.
    env: str = ""
    # What an empty value means, for the listing. Every one of these fields
    # falls back to something rather than to nothing, and "unset" on its own
    # invites setting it to a value it already behaves as.
    empty: str = "unset"


SETTINGS = (
    Setting("model", "model", "model", _text,
            "the model driving the agent", "a model name the LLM server lists",
            env="DEFAULT_LLM_MODEL",
            empty="unset (the first model the server lists)"),
    Setting("llm-host", "llm_host", "llmHost", _text,
            "the OpenAI-compatible server the models are called on",
            "a URL, e.g. http://localhost:11434",
            empty="unset ($LLM_HOST, or http://localhost:11434)"),
    Setting("llm-timeout", "llm_timeout_s", "llmTimeoutS", _seconds(1),
            "per-request timeout for calls to the LLM server, in seconds",
            "a number of seconds, at least 1"),
    Setting("max-steps", "max_steps", "maxSteps", _whole(1),
            "hard cap on agent loop iterations", "a whole number, at least 1"),
    Setting("subagent-model", "subagent_model", "subagentModel", _text,
            "the model subagents run on (unset = the same one)",
            "a model name, or reset to use the main one",
            empty="unset (the main model)"),
    Setting("subagent-max-steps", "subagent_max_steps", "subagentMaxSteps", _whole(1),
            "step cap for one subagent, separate from max-steps",
            "a whole number, at least 1"),
    Setting("parse-intent", "parse_intent", "parseIntent", _flag,
            "parse the task into structured intent before acting",
            "on or off"),
    Setting("intent-model", "intent_model", "intentModel", _text,
            "the model intent parsing runs on (unset = the same one)",
            "a model name, or reset to use the main one",
            empty="unset (the main model)"),
    Setting("compact-model", "compact_model", "compactModel", _text,
            "the model history compaction runs on (unset = the same one)",
            "a model name, or reset to use the main one",
            empty="unset (the main model)"),
    Setting("compact-keep-last", "compact_keep_last", "compactKeepLast", _whole(2),
            "how many recent messages survive compaction verbatim",
            "a whole number, at least 2"),
    Setting("context-char-budget", "context_char_budget", CONTEXT_CHAR_BUDGET_KEY,
            _whole(MIN_CONTEXT_CHAR_BUDGET),
            "characters of history allowed before it is compacted",
            f"a whole number, at least {MIN_CONTEXT_CHAR_BUDGET:,}, or the history "
            "compacts on every step"),
    Setting("embedding-model", "embedding_model", "embeddingModel", _embedding_backend,
            "embedding backend ranking search_tools against deferred MCP tools",
            'a model name, or "off" for plain keyword matching',
            empty="off (plain keyword matching)"),
    Setting("max-output-chars", "max_output_chars", "maxOutputChars", _whole(500),
            "where one tool's output is truncated before the model sees it",
            "a whole number, at least 500", scope="run"),
    Setting("shell-timeout", "shell_timeout_s", "shellTimeoutS", _whole(1),
            "how long one run_shell command may take, in seconds",
            "a whole number of seconds, at least 1", scope="run"),
    Setting("mcp-connect-timeout", "mcp_connect_timeout_s", "mcpConnectTimeoutS", _seconds(1),
            "how long one MCP server gets to finish its handshake, in seconds",
            "a number of seconds, at least 1", scope="run"),
    Setting("system-prompt", "system_prompt", SYSTEM_PROMPT_KEY, _prose,
            "the system prompt, replacing the built-in one",
            "some text, or: file <path>", scope="session", from_file=True,
            empty="the built-in prompt"),
    Setting("theme-color", "theme_color", THEME_COLOR_KEY, normalize_hex_color,
            "the UI accent colour", "a 6-digit hex colour, e.g. #00b4d8",
            empty="the built-in accent"),
)

SETTINGS_BY_NAME = {s.name: s for s in SETTINGS}


def find_setting(name: str) -> Setting:
    """The setting `name` refers to, or None.

    Underscored spellings resolve to the hyphenated ones: every name here is
    read off the AgentConfig field it sets, which is underscored, so that is
    at least as likely to be typed. "chart" for "char" is the typo the
    budget's name invites, and costs one line to forgive."""
    key = (name or "").strip().lower().lstrip("/").replace("_", "-")
    return SETTINGS_BY_NAME.get(key.replace("chart", "char"))


def env_setting(setting: Setting):
    """A setting's value from its environment variable, or None.

    Read when it's needed rather than at import, so exporting the variable
    and running the agent in the same shell does what it looks like it
    does — and so a test can set it."""
    if not setting.env:
        return None
    return setting.parse(os.environ.get(setting.env, ""))


def saved_setting(setting: Setting, path: str = None):
    """The saved value of one setting, or None if nothing usable is saved.

    Junk reads as nothing rather than propagating: a hand-edited settings
    file must not be able to put the loop into permanent compaction, paint
    the UI an unparseable colour, or cap the agent at zero steps."""
    data = load_settings(path)
    if setting.key not in data:
        return None
    return setting.parse(data[setting.key])


def saved_settings(path: str = None) -> dict:
    """Every usable saved preference, as {field: value} — ready to be merged
    into an AgentConfig."""
    values = {}
    for setting in SETTINGS:
        value = saved_setting(setting, path)
        if value is not None:
            values[setting.field] = value
    return values


@dataclass
class AgentConfig:
    # Empty on purpose: there is no model name compiled in here, because any
    # name picked as a default is the wrong one on every install that doesn't
    # happen to run it. The CLI fills this in from --model, from what /model
    # saved, or by asking the server what it has; if none of those produce a
    # name it refuses to start, which is a better answer than a first call
    # that fails somewhere in the server with "model not found".
    model: str = ""
    llm_host: str = ""   # empty = use LLM_HOST env var or http://localhost:11434
    llm_api_key: str = ""  # empty = use LLM_API_KEY env var; never hardcode this

    # All file/shell operations are confined to this directory. Paths that
    # resolve outside it are rejected before any tool runs.
    project_root: str = "."

    # Tool names that execute WITHOUT asking for human approval first.
    # Anything not listed here (write_file, edit_file, run_shell by default)
    # will print what it's about to do and wait for confirmation, unless
    # auto_approve=True.
    safe_tools: tuple = (
        "read_file", "list_dir", "search_files", "glob_files",
        "git_diff", "git_status", "git_log", "git_show", "git_branch", "git_fetch",
        "save_memory", "search_tools",
        "ask_user",     # asking the person a question changes nothing
        "spawn_agent",  # spawning one changes nothing either — whatever the
                         # subagent then does is approved on its own terms
        "list_resources", "read_resource",   # MCP Resources capability — read-only
    )

    auto_approve: bool = False        # True = never prompt (use in CI with care)
    max_steps: int = 100              # hard cap on agent loop iterations

    # Project-local file of durable notes (conventions, gotchas, preferences)
    # the agent has chosen to remember via the save_memory tool. Resolved
    # relative to project_root, not CWD. Read back in and folded into the
    # system prompt at the start of every new (non-resumed) session.
    memory_path: str = "agent_memory.md"

    # Replaces the built-in system prompt (agent.SYSTEM_PROMPT) when set.
    # Empty = use the built-in one. Set from --system-prompt, or read from a
    # file with --system-prompt-file. Note the built-in prompt is what tells
    # the model the tool discipline the loop relies on (prefer edit_file over
    # write_file, finish with plain text rather than a tool call, save_memory
    # for durable facts) — a replacement should cover the same ground.
    system_prompt: str = ""

    # Subagents (the spawn_agent tool). Empty/zero = same as the parent's.
    # A separate step budget matters: a runaway subagent would otherwise eat
    # the whole run's allowance.
    subagent_model: str = ""
    subagent_max_steps: int = 40

    # Accent colour for the UI, as #rrggbb. Empty = the built-in one.
    theme_color: str = ""

    # Parse the freeform task into structured intent (task_type, target_files,
    # constraints, risk_level) before the agent starts acting.
    parse_intent: bool = True
    intent_model: str = ""            # empty = reuse `model` for intent parsing too
    max_retries: int = 3              # retries per model call on bad/malformed output
    llm_timeout_s: float = 300.0      # per-request timeout for chat/intent/compaction calls to the LLM server
    shell_timeout_s: int = 30
    max_output_chars: int = 8000      # truncate tool output before feeding back to model
    context_char_budget: int = 200_000  # rough trim threshold (chars, not tokens)

    # When context_char_budget is exceeded, the history is compacted: an LLM
    # call summarizes everything except the system+task messages and the most
    # recent `compact_keep_last` messages, which are kept verbatim. Falls back
    # to the old drop-oldest trim if the summarization call itself fails.
    compact_keep_last: int = 20
    compact_model: str = ""           # empty = reuse `model` for compaction too
    log_path: str = "agent_run.log"
    # stderr from every stdio-transport MCP server (built-in + custom) is
    # redirected here instead of the terminal, so a chatty/crashing server
    # doesn't interleave raw debug output with the Rich UI.
    mcp_log_path: str = "mcp_servers.log"
    db_path: str = "agent_sessions.db"  # SQLite file storing session/message history

    # How long one MCP server gets to complete its handshake before the
    # session gives up on it and carries on without it. A server that starts
    # but never speaks MCP (misconfigured, waiting on a lock, an unreachable
    # URL) would otherwise hold up startup indefinitely.
    mcp_connect_timeout_s: float = 20.0

    # Optional path to a Claude-Desktop-style MCP config file
    # ({"mcpServers": {"name": {"command": ..., "args": [...], "env": {...}}}})
    # for adding extra tool servers beyond the built-in one. Empty = none.
    mcp_config_path: str = ""

    # Extra MCP servers specified directly (e.g. via repeatable --mcp-server
    # CLI flags), as {name: {"command": ..., "args": [...], "env": {...}}}.
    # Merged with mcp_config_path's servers; wins on a name clash.
    mcp_servers: dict = field(default_factory=dict)

    # Embedding backend for ranking search_tools queries against deferred
    # MCP tool descriptions:
    #   "nomic-local" (default) - on-device via the `nomic` package, no
    #       server involved (needs `pip install "nomic[local]"`; the model
    #       itself downloads on first use)
    #   any other string - a remote OpenAI-compatible embedding model name
    #       (e.g. "mxbai-embed-large"), fetched from llm_host/llm_api_key
    #   "" - disabled; search_tools falls back to plain keyword matching
    # Falls back to keyword matching automatically, per call, if the
    # configured backend errors (missing dependency, model not pulled,
    # network error).
    embedding_model: str = "nomic-local"

    # Commands the agent is never allowed to run, regardless of approval.
    # A footgun guard, NOT a security boundary — plain substring matching,
    # trivially sidestepped by a variant spelling (see Tools.run_shell).
    # Real isolation has to come from the OS (container/VM).
    denied_shell_patterns: tuple = field(default_factory=lambda: (
        "rm -rf /", "rm -rf /*", ":(){ :|:& };:", "mkfs", "dd if=",
        "> /dev/sda", "shutdown", "reboot", "sudo ", "curl | sh", "wget | sh",
    ))

    # ---- tool-side knobs -> built-in MCP server subprocess ----
    #
    # The tools themselves run inside the built-in MCP server, a separate
    # process (see mcp_server.py). Anything below that governs tool
    # behaviour rather than the agent loop — the shell timeout, output
    # truncation, the memory file, the shell denylist — therefore has to be
    # handed across that process boundary explicitly, or the server falls
    # back to these defaults and silently ignores whatever the caller
    # configured. These two methods are the one place that mapping lives.

    def tool_server_env(self) -> dict:
        """Env vars carrying the tool-side knobs to the built-in server."""
        return {
            "AGENT_PROJECT_ROOT": self.project_root,
            "AGENT_SHELL_TIMEOUT_S": str(self.shell_timeout_s),
            "AGENT_MAX_OUTPUT_CHARS": str(self.max_output_chars),
            "AGENT_MEMORY_PATH": self.memory_path,
            "AGENT_DENIED_SHELL_PATTERNS": json.dumps(list(self.denied_shell_patterns)),
        }

    @classmethod
    def from_tool_server_env(cls, env: dict = None) -> "AgentConfig":
        """Rebuild the tool-relevant slice of a config inside the built-in
        server process, from what tool_server_env() exported. Every field
        falls back to this class's default if the variable is missing or
        unparseable — a malformed value must not stop the server from
        starting, and the defaults are the same ones the agent assumes."""
        env = os.environ if env is None else env
        defaults = cls()

        def _num(name, cast, default):
            raw = env.get(name)
            if raw is None:
                return default
            try:
                return cast(raw)
            except (TypeError, ValueError):
                return default

        denied = defaults.denied_shell_patterns
        raw_denied = env.get("AGENT_DENIED_SHELL_PATTERNS")
        if raw_denied is not None:
            try:
                parsed = json.loads(raw_denied)
                if isinstance(parsed, list):
                    denied = tuple(str(x) for x in parsed)
            except json.JSONDecodeError:
                pass

        return cls(
            project_root=env.get("AGENT_PROJECT_ROOT", defaults.project_root),
            shell_timeout_s=_num("AGENT_SHELL_TIMEOUT_S", lambda v: int(float(v)), defaults.shell_timeout_s),
            max_output_chars=_num("AGENT_MAX_OUTPUT_CHARS", int, defaults.max_output_chars),
            memory_path=env.get("AGENT_MEMORY_PATH", defaults.memory_path),
            denied_shell_patterns=denied,
        )
