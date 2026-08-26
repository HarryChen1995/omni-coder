"""AgentConfig — defaults that encode policy, so a silent change to any of
them is a behaviour change worth catching."""

import dataclasses
import json

import pytest

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


def test_mcp_connect_timeout_is_bounded_by_default():
    """A server that never completes its handshake must not hang the session."""
    assert AgentConfig().mcp_connect_timeout_s == 20.0


def test_system_prompt_defaults_to_empty_meaning_built_in():
    assert AgentConfig().system_prompt == ""


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


# ---------------- tool_server_env round trip ----------------
#
# The tools run in a separate process (mcp_server.py), so anything below
# that governs tool behaviour has to cross that boundary explicitly. These
# pin the mapping in both directions: a knob that stops being exported is a
# knob the built-in server silently ignores.

def test_tool_server_env_exports_every_tool_side_knob():
    cfg = AgentConfig(project_root="/repo", shell_timeout_s=7, max_output_chars=99,
                      memory_path="notes/mem.md", denied_shell_patterns=("boom",))
    env = cfg.tool_server_env()
    assert env["AGENT_PROJECT_ROOT"] == "/repo"
    assert env["AGENT_SHELL_TIMEOUT_S"] == "7"
    assert env["AGENT_MAX_OUTPUT_CHARS"] == "99"
    assert env["AGENT_MEMORY_PATH"] == "notes/mem.md"
    assert json.loads(env["AGENT_DENIED_SHELL_PATTERNS"]) == ["boom"]
    assert all(isinstance(v, str) for v in env.values())   # subprocess env must be strings


def test_from_tool_server_env_restores_what_was_exported():
    cfg = AgentConfig(project_root="/repo", shell_timeout_s=7, max_output_chars=99,
                      memory_path="notes/mem.md", denied_shell_patterns=("boom", "kaboom"))
    restored = AgentConfig.from_tool_server_env(cfg.tool_server_env())
    assert restored.project_root == "/repo"
    assert restored.shell_timeout_s == 7
    assert restored.max_output_chars == 99
    assert restored.memory_path == "notes/mem.md"
    assert restored.denied_shell_patterns == ("boom", "kaboom")


def test_from_tool_server_env_falls_back_to_defaults_when_empty():
    restored = AgentConfig.from_tool_server_env({})
    defaults = AgentConfig()
    assert restored.project_root == defaults.project_root
    assert restored.shell_timeout_s == defaults.shell_timeout_s
    assert restored.denied_shell_patterns == defaults.denied_shell_patterns


@pytest.mark.parametrize("env", [
    {"AGENT_SHELL_TIMEOUT_S": "not-a-number"},
    {"AGENT_MAX_OUTPUT_CHARS": ""},
    {"AGENT_DENIED_SHELL_PATTERNS": "{not json"},
    {"AGENT_DENIED_SHELL_PATTERNS": '"a string, not a list"'},
])
def test_from_tool_server_env_ignores_malformed_values(env):
    """A garbled variable must not stop the tool server from starting."""
    restored = AgentConfig.from_tool_server_env(env)
    defaults = AgentConfig()
    assert restored.shell_timeout_s == defaults.shell_timeout_s
    assert restored.max_output_chars == defaults.max_output_chars
    assert restored.denied_shell_patterns == defaults.denied_shell_patterns


def test_from_tool_server_env_reads_os_environ_by_default(monkeypatch):
    monkeypatch.setenv("AGENT_PROJECT_ROOT", "/from/environ")
    monkeypatch.setenv("AGENT_SHELL_TIMEOUT_S", "3")
    restored = AgentConfig.from_tool_server_env()
    assert restored.project_root == "/from/environ" and restored.shell_timeout_s == 3
