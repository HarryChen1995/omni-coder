"""AgentConfig — defaults that encode policy, so a silent change to any of
them is a behaviour change worth catching."""

import dataclasses
import json

import pytest

from omni import config
from omni.config import AgentConfig


def test_defaults_are_conservative():
    cfg = AgentConfig()
    assert cfg.model == ""                  # no model name is compiled in; ask the server
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
    assert cfg.context_window_budget == 50_000
    assert cfg.compact_keep_last == 20
    assert cfg.max_retries == 3


def test_read_only_tools_are_safe_and_writes_are_not():
    safe = set(AgentConfig().safe_tools)
    for tool in ("read_file", "list_dir", "search_files", "glob_files",
                 "git_diff", "git_status", "git_log", "git_show", "git_branch",
                 "git_fetch", "save_memory", "search_tools", "ask_user",
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


# ---------------- the settings file ----------------

@pytest.fixture
def settings(tmp_path):
    return str(tmp_path / "omni-coder-settings.json")


def _saved(name, path):
    """One setting's saved value, by the name its slash command goes by."""
    return config.saved_setting(config.SETTINGS_BY_NAME[name], path)


@pytest.mark.parametrize("given,expected", [
    ("#00b4d8", "#00b4d8"),
    ("00b4d8", "#00b4d8"),       # the "#" is optional — shells eat it
    ("  #00B4D8  ", "#00B4D8"),
])
def test_a_hex_colour_is_normalized(given, expected):
    assert config.normalize_hex_color(given) == expected


@pytest.mark.parametrize("bad", ["blue", "#12345", "#1234567", "#gggggg", "", None, "#abc"])
def test_anything_that_is_not_six_hex_digits_is_refused(bad):
    """Including "#abc": three-digit shorthand is as likely to be a typo, and
    guessing wrong silently recolours the whole UI."""
    assert config.normalize_hex_color(bad) is None


def test_a_setting_round_trips(settings):
    config.save_setting(config.THEME_COLOR_KEY, "#00b4d8", path=settings)
    assert _saved("theme-color", settings) == "#00b4d8"


def test_saving_a_setting_leaves_the_mcp_servers_alone(settings):
    """The same file holds registered MCP servers — a preference written here
    must not cost the user their servers."""
    with open(settings, "w") as f:
        json.dump({"mcpServers": {"docs": {"command": "node"}}}, f)
    config.save_setting(config.THEME_COLOR_KEY, "#00b4d8", path=settings)
    data = json.load(open(settings))
    assert data["mcpServers"] == {"docs": {"command": "node"}}
    assert data[config.THEME_COLOR_KEY] == "#00b4d8"


def test_a_setting_can_be_cleared(settings):
    config.save_setting(config.THEME_COLOR_KEY, "#00b4d8", path=settings)
    config.save_setting(config.THEME_COLOR_KEY, None, path=settings)
    assert _saved("theme-color", settings) is None
    assert config.THEME_COLOR_KEY not in json.load(open(settings))


def test_a_missing_or_corrupt_settings_file_reads_as_no_settings(tmp_path):
    """A broken settings file falls back to defaults rather than stopping the
    agent from starting."""
    assert config.load_settings(str(tmp_path / "nope.json")) == {}
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert config.load_settings(str(corrupt)) == {}
    assert _saved("theme-color", str(corrupt)) is None
    listy = tmp_path / "list.json"
    listy.write_text("[1, 2]")
    assert config.load_settings(str(listy)) == {}


def test_a_hand_edited_junk_colour_does_not_reach_the_ui(settings):
    with open(settings, "w") as f:
        json.dump({config.THEME_COLOR_KEY: "chartreuse"}, f)
    assert _saved("theme-color", settings) is None


@pytest.mark.parametrize("given,expected", [
    (200_000, 200_000),
    ("200000", 200_000),
    ("200_000", 200_000),        # the separators the default is written with
    ("200,000", 200_000),
    ("  50000 ", 50_000),
])
def test_a_token_budget_is_normalized(given, expected):
    assert config.SETTINGS_BY_NAME["context-window-budget"].parse(given) == expected


@pytest.mark.parametrize("bad", ["", None, "lots", "-5", "1e5", "20.5", True, "499"])
def test_anything_that_is_not_a_usable_budget_is_refused(bad):
    """Including 499: under the floor the history compacts on every step and
    spends a summarization call per turn to save nothing."""
    assert config.SETTINGS_BY_NAME["context-window-budget"].parse(bad) is None


def test_a_system_prompt_round_trips(settings):
    config.save_setting(config.SYSTEM_PROMPT_KEY, "Be terse.", path=settings)
    assert _saved("system-prompt", settings) == "Be terse."


def test_a_blank_saved_system_prompt_reads_as_none(settings):
    """A prompt of blank lines would replace the built-in one with nothing,
    which reads as the agent forgetting how to use its own tools."""
    config.save_setting(config.SYSTEM_PROMPT_KEY, "  \n ", path=settings)
    assert _saved("system-prompt", settings) is None
    config.save_setting(config.SYSTEM_PROMPT_KEY, 17, path=settings)
    assert _saved("system-prompt", settings) is None


def test_a_char_budget_round_trips_and_clears(settings):
    config.save_setting(config.CONTEXT_WINDOW_BUDGET_KEY, 50_000, path=settings)
    assert _saved("context-window-budget", settings) == 50_000
    config.save_setting(config.CONTEXT_WINDOW_BUDGET_KEY, None, path=settings)
    assert _saved("context-window-budget", settings) is None


def test_a_hand_edited_junk_budget_does_not_reach_the_loop(settings):
    """Nothing usable saved is the same as nothing saved — a hand-edited file
    shouldn't be able to put the loop into permanent compaction."""
    with open(settings, "w") as f:
        json.dump({config.CONTEXT_WINDOW_BUDGET_KEY: 12}, f)
    assert _saved("context-window-budget", settings) is None


# ---------------- the settings registry ----------------

def test_every_setting_names_a_real_config_field_and_a_distinct_key():
    fields = {f.name for f in dataclasses.fields(AgentConfig)}
    keys, names = set(), set()
    for setting in config.SETTINGS:
        assert setting.field in fields, setting.name
        assert setting.key not in keys and setting.name not in names
        assert setting.scope in ("now", "session", "run")
        keys.add(setting.key)
        names.add(setting.name)


def test_every_settings_default_survives_its_own_parser():
    """A default the setting would refuse if it were typed means /<name> reset
    lands on a value /<name> <that value> won't accept."""
    for setting in config.SETTINGS:
        default = getattr(AgentConfig, setting.field)
        if default in ("", 0):
            continue          # unset is a state, not a value you can type
        assert setting.parse(default) == default, setting.name


def test_secrets_and_footguns_are_not_saveable():
    """--auto-approve outliving the run that wanted it is a footgun, and a
    key in a plaintext settings file is a leak — neither is a preference."""
    saveable = {s.field for s in config.SETTINGS}
    assert "auto_approve" not in saveable and "llm_api_key" not in saveable
    assert not saveable & {"project_root", "db_path", "log_path", "mcp_log_path"}


@pytest.mark.parametrize("typed,expected", [
    ("/max-steps", "max-steps"),
    ("max_steps", "max-steps"),
    ("/SYSTEM_PROMPT", "system-prompt"),
    ("/context_window_budget", "context-window-budget"),   # underscores instead of hyphens
    ("CONTEXT-WINDOW-BUDGET", "context-window-budget"),  # case is ignored
])
def test_a_setting_is_found_however_its_name_is_spelled(typed, expected):
    assert config.find_setting(typed).name == expected


@pytest.mark.parametrize("typed", ["", None, "/nope", "compact"])
def test_anything_else_is_not_a_setting(typed):
    assert config.find_setting(typed) is None


@pytest.mark.parametrize("given,expected", [("on", True), ("off", False), ("yes", True),
                                            ("0", False), ("TRUE", True)])
def test_a_flag_setting_reads_how_it_was_typed(given, expected):
    assert config.SETTINGS_BY_NAME["parse-intent"].parse(given) is expected


def test_an_unrecognized_word_is_not_a_flag():
    """It must not quietly read as False — "parse-intent maybe" would then
    silently turn intent parsing off."""
    assert config.SETTINGS_BY_NAME["parse-intent"].parse("maybe") is None


def test_the_embedding_backend_can_be_turned_off_by_name():
    """"" can be passed to the flag but not typed at a slash command."""
    assert config.SETTINGS_BY_NAME["embedding-model"].parse("off") == ""
    assert config.SETTINGS_BY_NAME["embedding-model"].parse("mxbai") == "mxbai"


def test_a_system_prompt_from_a_file_keeps_its_formatting():
    """Unlike the one-line settings it is not stripped: the trailing newline
    and the indentation are the user's."""
    assert config.SETTINGS_BY_NAME["system-prompt"].parse("A.\n\n  B.\n") == "A.\n\n  B.\n"
    assert config.SETTINGS_BY_NAME["system-prompt"].parse(" \n ") is None


def test_saved_settings_collects_every_usable_value_and_skips_junk(settings):
    with open(settings, "w") as f:
        json.dump({"maxSteps": 7, "themeColor": "chartreuse", "parseIntent": False,
                   "mcpServers": {"docs": {}}}, f)
    assert config.saved_settings(settings) == {"max_steps": 7, "parse_intent": False}
