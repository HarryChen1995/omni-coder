"""MCP server exposing the coding tools, each scoped to a project root.

Run standalone to test with any MCP client:
    AGENT_PROJECT_ROOT=/path/to/repo python -m omni.mcp_server

The rest of the tool-side policy comes from the environment too
(AGENT_SHELL_TIMEOUT_S, AGENT_MAX_OUTPUT_CHARS, AGENT_MEMORY_PATH,
AGENT_DENIED_SHELL_PATTERNS) — see AgentConfig.tool_server_env().

Any MCP-compatible client (not just this agent) can now use these tools —
Claude Desktop, another agent framework, etc. — all sharing the same
project-scope/approval-preview logic in tools.py.
"""

from mcp.server.fastmcp import FastMCP

from .config import AgentConfig
from .tools import Tools

# The tool-side knobs (project root, shell timeout, output truncation,
# memory file, shell denylist) come from the environment the client set up
# for this subprocess — see AgentConfig.tool_server_env(). Anything absent
# falls back to AgentConfig's own default, so running this module by hand
# (`AGENT_PROJECT_ROOT=/repo python -m omni.mcp_server`) still works.
cfg = AgentConfig.from_tool_server_env()
impl = Tools(cfg)

mcp = FastMCP("omni-tools")


# ---- Tools exposed to any MCP client (these are what the LLM sees) ----

@mcp.tool()
def read_file(path: str, start_line: int = None, end_line: int = None) -> str:
    """Read a file, optionally a specific line range."""
    return impl.read_file(path, start_line, end_line)


@mcp.tool()
def list_dir(path: str = ".") -> str:
    """List files in a directory."""
    return impl.list_dir(path)


@mcp.tool()
def search_files(pattern: str, path: str = ".", glob: str = None, case_insensitive: bool = False) -> str:
    """Regex search across files under a path, skipping noise directories
    (.git, node_modules, __pycache__, build output, etc.). Optionally
    restrict to filenames matching `glob` (e.g. "*.py")."""
    return impl.search_files(pattern, path, glob, case_insensitive)


@mcp.tool()
def glob_files(pattern: str, path: str = ".") -> str:
    """Find files by glob pattern, e.g. "**/*.tsx" or "src/**/test_*.py".
    Results are sorted newest-first. Use this to discover files by name/
    location; use search_files to find files by content."""
    return impl.glob_files(pattern, path)


@mcp.tool()
def git_diff(path: str = ".") -> str:
    """Show uncommitted git changes in the project."""
    return impl.git_diff(path)


@mcp.tool()
def git_status(path: str = ".") -> str:
    """Show the working tree status (short format) and current branch."""
    return impl.git_status(path)


@mcp.tool()
def git_log(path: str = ".", max_count: int = 20) -> str:
    """Show recent commit history (hash, date, author, subject)."""
    return impl.git_log(path, max_count)


@mcp.tool()
def git_show(ref: str = "HEAD", path: str = ".") -> str:
    """Show a commit's metadata and diff (defaults to HEAD)."""
    return impl.git_show(ref, path)


@mcp.tool()
def git_branch(path: str = ".") -> str:
    """List local branches, marking the current one and its upstream tracking info."""
    return impl.git_branch(path)


@mcp.tool()
def git_fetch(remote: str = "origin") -> str:
    """Update remote-tracking refs from a remote without touching the working tree."""
    return impl.git_fetch(remote)


@mcp.tool()
def save_memory(note: str) -> str:
    """Persist a short, durable note to this project's memory file — a \
convention, gotcha, or preference worth remembering — so it's automatically \
recalled at the start of every future session. Don't use it for task-specific \
status or anything derivable from the code itself."""
    return impl.save_memory(note)


@mcp.tool()
def git_add(paths: str = ".") -> str:
    """Stage files for commit. `paths` is a space-separated list of file \
paths relative to the project root, or "." to stage all changes."""
    return impl.git_add(paths)


@mcp.tool()
def git_commit(message: str) -> str:
    """Commit staged changes with the given commit message."""
    return impl.git_commit(message)


@mcp.tool()
def git_pull(remote: str = "origin", branch: str = "") -> str:
    """Fetch and merge from a remote into the current branch."""
    return impl.git_pull(remote, branch)


@mcp.tool()
def git_push(remote: str = "origin", branch: str = "") -> str:
    """Push commits to a remote (defaults to pushing the current branch to origin)."""
    return impl.git_push(remote, branch)


@mcp.tool()
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Create a NEW file with content. Fails if the file already exists
    unless overwrite=true. Use edit_file for existing files."""
    return impl.write_file(path, content, overwrite)


@mcp.tool()
def edit_file(path: str, old_str: str, new_str: str) -> str:
    """Replace an exact, unique block of text in an existing file. old_str
    must match precisely (include enough context to be unique)."""
    return impl.edit_file(path, old_str, new_str)


@mcp.tool()
def run_shell(command: str) -> str:
    """Run a shell command in the project root. A few catastrophic command
    patterns are refused outright; everything else runs, so the caller (and
    its human) is the real check on what this does."""
    return impl.run_shell(command)


# ---- Internal tools: dry-run previews for the approval UI + existence
# checks for intent validation. Named with a leading underscore so the
# client can filter them out of what it hands to the LLM, while still
# calling them directly for its own approval-flow logic. ----

@mcp.tool()
def _preview_edit(path: str, old_str: str, new_str: str) -> str:
    """Internal: dry-run diff preview for edit_file, no write performed."""
    ok, msg = impl.preview_edit(path, old_str, new_str)
    return f"OK\n{msg}" if ok else f"ERROR\n{msg}"


@mcp.tool()
def _preview_write(path: str, content: str, overwrite: bool = False) -> str:
    """Internal: dry-run preview for write_file, no write performed."""
    is_new, preview = impl.preview_write(path, content, overwrite)
    return f"{'NEW' if is_new else 'DIFF'}\n{preview}"


@mcp.tool()
def _file_exists(path: str) -> str:
    """Internal: existence check confined to the project root, for intent validation."""
    return "true" if impl.file_exists(path) else "false"


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
