"""AgentConfig — defaults that encode policy, so a silent change to any of
them is a behaviour change worth catching."""

import dataclasses

from omni.config import AgentConfig


def test_defaults_are_conservative():
    cfg = AgentConfig()
    assert cfg.auto_approve is False        # never skip approval unasked
    assert cfg.parse_intent is True
    assert cfg.project_root == "."
    assert cfg.max_steps == 100


def test_llm_connection_defaults_are_empty_so_env_wins():
    """Empty means "fall through to $LLM_HOST / $LLM_API_KEY" in llm_client."""
    cfg = AgentConfig()
    assert cfg.llm_host == "" and cfg.llm_api_key == ""


def test_timeouts_and_budgets():
    cfg = AgentConfig()
    assert cfg.llm_timeout_s == 300.0
    assert cfg.shell_timeout_s == 30
    assert cfg.max_output_chars == 8000
    assert cfg.context_char_budget == 200_000
    assert cfg.compact_keep_last == 20
    assert cfg.max_retries == 3


def test_read_only_tools_are_safe_and_writes_are_not():
    safe = set(AgentConfig().safe_tools)
    for tool in ("read_file", "list_dir", "search_files", "glob_files",
                 "git_diff", "git_status", "git_log", "git_show", "git_branch",
                 "git_fetch", "save_memory", "search_tools",
                 "list_resources", "read_resource"):
        assert tool in safe, f"{tool} should run without approval"
    for tool in ("write_file", "edit_file", "run_shell",
                 "git_add", "git_commit", "git_pull", "git_push"):
        assert tool not in safe, f"{tool} must require approval"


def test_denied_shell_patterns_cover_the_obvious_footguns():
    denied = AgentConfig().denied_shell_patterns
    for pattern in ("rm -rf /", "mkfs", "dd if=", "sudo ", "shutdown"):
        assert any(pattern in d for d in denied), f"{pattern} not blocked"


def test_mutable_defaults_are_not_shared_between_instances():
    """A shared dict/list default would leak MCP servers between configs."""
    a, b = AgentConfig(), AgentConfig()
    a.mcp_servers["x"] = {"command": "y"}
    assert b.mcp_servers == {}


def test_paths_have_sensible_relative_defaults():
    cfg = AgentConfig()
    assert cfg.log_path == "agent_run.log"
    assert cfg.db_path == "agent_sessions.db"
    assert cfg.memory_path == "agent_memory.md"
    assert cfg.mcp_log_path.endswith(".log")


def test_optional_model_overrides_default_to_empty():
    """Empty means "reuse `model`" for each of these."""
    cfg = AgentConfig()
    assert cfg.intent_model == "" and cfg.compact_model == ""


def test_embedding_model_defaults_to_on_device():
    assert AgentConfig().embedding_model == "nomic-local"


def test_every_field_is_overridable():
    fields = {f.name for f in dataclasses.fields(AgentConfig)}
    cfg = AgentConfig(model="m", project_root="/p", max_steps=1, auto_approve=True)
    assert cfg.model == "m" and cfg.max_steps == 1 and cfg.auto_approve is True
    assert "safe_tools" in fields and "denied_shell_patterns" in fields
