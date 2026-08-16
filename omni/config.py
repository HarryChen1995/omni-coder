"""Configuration for the coding agent."""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    model: str = "qwen3.6:35b"
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
        "list_resources", "read_resource",   # MCP Resources capability — read-only
    )

    auto_approve: bool = False        # True = never prompt (use in CI with care)
    max_steps: int = 100              # hard cap on agent loop iterations

    # Project-local file of durable notes (conventions, gotchas, preferences)
    # the agent has chosen to remember via the save_memory tool. Resolved
    # relative to project_root, not CWD. Read back in and folded into the
    # system prompt at the start of every new (non-resumed) session.
    memory_path: str = "agent_memory.md"

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
    denied_shell_patterns: tuple = field(default_factory=lambda: (
        "rm -rf /", "rm -rf /*", ":(){ :|:& };:", "mkfs", "dd if=",
        "> /dev/sda", "shutdown", "reboot", "sudo ", "curl | sh", "wget | sh",
    ))
