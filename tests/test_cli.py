"""cli.py — the Typer surface (flags, one-shot runs, MCP registry commands,
session management) and the REPL's helper functions.

CodingAgent.run and the MCP client are mocked, so nothing spawns a
subprocess or reaches a model; HOME is redirected so the real
~/.omni-coder settings file is never touched.
"""

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from omni import cli as cli_mod
from omni.cli import app
from omni.config import AgentConfig
from omni.llm_client import LLMError

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_home(mocker, tmp_path):
    """Point ~ at tmp_path so the global settings file stays untouched."""
    home = tmp_path / "home"
    (home / ".omni-coder").mkdir(parents=True)
    mocker.patch("os.path.expanduser", lambda p: p.replace("~", str(home)))
    return home


@pytest.fixture
def settings_path(isolated_home):
    return isolated_home / ".omni-coder" / "omni-coder-settings.json"


@pytest.fixture
def no_interactive(mocker):
    """Guard: a bare `omni` invocation must never open a real REPL in tests."""
    return mocker.patch.object(cli_mod, "_interactive", mocker.AsyncMock())


def invoke(*args, **kwargs):
    return runner.invoke(app, list(args), **kwargs)


# ---------------- one-shot run ----------------

def test_one_shot_run_prints_the_result(mocker, tmp_path):
    run = mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="THE RESULT"))
    r = invoke("do the thing", "--db-path", str(tmp_path / "s.db"),
               "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 0 and "THE RESULT" in r.output
    assert run.await_args.args[0] == "do the thing"


def test_flags_are_threaded_into_agentconfig(mocker, tmp_path):
    captured = {}
    mocker.patch.object(cli_mod.CodingAgent, "__init__",
                        lambda self, cfg: captured.update(cfg=cfg))
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="ok"))
    invoke("t", "--project-root", str(tmp_path), "--model", "my-model",
           "--llm-host", "http://h:1", "--llm-api-key", "sk-k", "--llm-timeout", "42",
           "--max-steps", "7", "--auto-approve", "--skip-intent-parsing",
           "--intent-model", "small", "--compact-model", "tiny",
           "--compact-keep-last", "9", "--context-window-budget", "12340",
           "--embedding-model", "", "--db-path", str(tmp_path / "d.db"),
           "--log-path", str(tmp_path / "l.log"))
    cfg = captured["cfg"]
    assert cfg.model == "my-model" and cfg.llm_host == "http://h:1"
    assert cfg.llm_api_key == "sk-k" and cfg.llm_timeout_s == 42
    assert cfg.max_steps == 7 and cfg.auto_approve is True
    assert cfg.parse_intent is False and cfg.intent_model == "small"
    assert cfg.compact_model == "tiny" and cfg.compact_keep_last == 9
    assert cfg.context_window_budget == 12340 and cfg.embedding_model == ""


def test_embedding_model_defaults_when_flag_omitted(mocker, tmp_path):
    captured = {}
    mocker.patch.object(cli_mod.CodingAgent, "__init__", lambda self, cfg: captured.update(cfg=cfg))
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="ok"))
    invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].embedding_model == AgentConfig.embedding_model


def test_safe_tool_flag_extends_the_auto_approved_set(mocker, tmp_path):
    captured = {}
    mocker.patch.object(cli_mod.CodingAgent, "__init__", lambda self, cfg: captured.update(cfg=cfg))
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="ok"))
    invoke("t", "--safe-tool", "docs__search", "--safe-tool", "docs__lookup",
           "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    safe = captured["cfg"].safe_tools
    assert "docs__search" in safe and "docs__lookup" in safe
    assert "read_file" in safe          # the built-in read-only set is kept


def test_defer_applies_to_inline_mcp_servers(mocker, tmp_path):
    captured = {}
    mocker.patch.object(cli_mod.CodingAgent, "__init__", lambda self, cfg: captured.update(cfg=cfg))
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="ok"))
    invoke("t", "--mcp-server", "docs=node srv.js", "--defer",
           "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].mcp_servers["docs"]["defer"] is True


def test_defer_without_any_server_says_so(mocker, tmp_path):
    """It used to be silently ignored unless paired with --add-mcp-server."""
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="ok"))
    r = invoke("t", "--defer", "--db-path", str(tmp_path / "d.db"),
               "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 0 and "--defer had no effect" in r.output


# ---------------- --system-prompt ----------------

def captured_cfg(mocker):
    captured = {}
    mocker.patch.object(cli_mod.CodingAgent, "__init__", lambda self, cfg: captured.update(cfg=cfg))
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(return_value="ok"))
    return captured


def test_system_prompt_flag_is_threaded_through(mocker, tmp_path):
    captured = captured_cfg(mocker)
    invoke("t", "--system-prompt", "Be terse.", "--db-path", str(tmp_path / "d.db"),
           "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].system_prompt == "Be terse."


def test_system_prompt_defaults_to_empty_meaning_built_in(mocker, tmp_path):
    captured = captured_cfg(mocker)
    invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].system_prompt == ""


def test_system_prompt_file_is_read(mocker, tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("You are a reviewer.\n\nBe specific.\n")
    captured = captured_cfg(mocker)
    invoke("t", "--system-prompt-file", str(prompt), "--db-path", str(tmp_path / "d.db"),
           "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].system_prompt == "You are a reviewer.\n\nBe specific.\n"


def test_both_system_prompt_flags_is_an_error(mocker, tmp_path):
    r = invoke("t", "--system-prompt", "a", "--system-prompt-file", str(tmp_path / "p.md"),
               "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1 and "not both" in r.output


def test_missing_system_prompt_file_is_an_error(mocker, tmp_path):
    r = invoke("t", "--system-prompt-file", str(tmp_path / "nope.md"),
               "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1 and "could not read" in r.output


def test_empty_system_prompt_file_is_an_error(mocker, tmp_path):
    """Falling back to the built-in prompt would look like the flag was
    ignored, which is worse than refusing."""
    empty = tmp_path / "empty.md"
    empty.write_text("\n  \n")
    r = invoke("t", "--system-prompt-file", str(empty), "--db-path", str(tmp_path / "d.db"),
               "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1 and "is empty" in r.output


def test_mcp_connect_timeout_flag_is_threaded_through(mocker, tmp_path):
    captured = captured_cfg(mocker)
    invoke("t", "--mcp-connect-timeout", "5", "--db-path", str(tmp_path / "d.db"),
           "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].mcp_connect_timeout_s == 5.0


def test_theme_colour_is_applied_and_recorded(mocker, tmp_path):
    accent = mocker.patch("omni.ui.set_accent")
    captured = captured_cfg(mocker)
    invoke("t", "--theme-color", "#00b4d8", "--db-path", str(tmp_path / "d.db"),
           "--log-path", str(tmp_path / "l.log"))
    accent.assert_called_once_with("#00b4d8")
    assert captured["cfg"].theme_color == "#00b4d8"


def test_theme_colour_accepts_a_bare_hex(mocker, tmp_path):
    accent = mocker.patch("omni.ui.set_accent")
    captured_cfg(mocker)
    invoke("t", "--theme-color", "00b4d8", "--db-path", str(tmp_path / "d.db"),
           "--log-path", str(tmp_path / "l.log"))
    accent.assert_called_once_with("#00b4d8")


@pytest.mark.parametrize("bad", ["blue", "#12345", "#gggggg", "#1234567"])
def test_a_bad_theme_colour_is_refused(mocker, tmp_path, bad):
    r = invoke("t", "--theme-color", bad, "--db-path", str(tmp_path / "d.db"),
               "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1 and "hex colour" in r.output


def test_no_theme_colour_leaves_the_default(mocker, tmp_path):
    accent = mocker.patch("omni.ui.set_accent")
    captured = captured_cfg(mocker)
    invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    accent.assert_not_called()
    assert captured["cfg"].theme_color == ""


def test_a_saved_theme_colour_is_used_when_the_flag_is_absent(mocker, tmp_path, settings_path):
    settings_path.write_text(json.dumps({"themeColor": "#00b4d8"}))
    accent = mocker.patch("omni.ui.set_accent")
    captured = captured_cfg(mocker)
    invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    accent.assert_called_once_with("#00b4d8")
    assert captured["cfg"].theme_color == "#00b4d8"


def test_the_flag_overrides_the_saved_colour_without_replacing_it(mocker, tmp_path, settings_path):
    """--theme-color is for trying a colour out. If it wrote itself back, one
    run with the flag would silently become the colour of every run after."""
    settings_path.write_text(json.dumps({"themeColor": "#00b4d8"}))
    accent = mocker.patch("omni.ui.set_accent")
    captured = captured_cfg(mocker)
    invoke("t", "--theme-color", "#ff0000", "--db-path", str(tmp_path / "d.db"),
           "--log-path", str(tmp_path / "l.log"))
    accent.assert_called_once_with("#ff0000")
    assert captured["cfg"].theme_color == "#ff0000"
    assert json.loads(settings_path.read_text())["themeColor"] == "#00b4d8"


# ---------------- /theme-color ----------------

def test_theme_color_command_saves_and_applies(mocker, settings_path):
    accent = mocker.patch("omni.ui.set_accent")
    tui = mocker.Mock()
    cfg = AgentConfig()
    cli_mod._theme_color_command(cfg, tui, "#00b4d8")
    accent.assert_called_once_with("#00b4d8")
    tui.restyle.assert_called_once()            # the app holds its own style map
    assert cfg.theme_color == "#00b4d8"
    assert json.loads(settings_path.read_text())["themeColor"] == "#00b4d8"


def test_theme_color_command_accepts_a_bare_hex(mocker, settings_path):
    mocker.patch("omni.ui.set_accent")
    cli_mod._theme_color_command(AgentConfig(), None, "00b4d8")
    assert json.loads(settings_path.read_text())["themeColor"] == "#00b4d8"


def test_theme_color_command_refuses_junk_and_saves_nothing(mocker, settings_path):
    accent = mocker.patch("omni.ui.set_accent")
    echoed = mocker.patch.object(cli_mod, "_echo")
    cfg = AgentConfig()
    cli_mod._theme_color_command(cfg, None, "chartreuse")
    accent.assert_not_called()
    assert cfg.theme_color == "" and not settings_path.exists()
    assert echoed.call_args.kwargs.get("err") is True
    assert "hex colour" in echoed.call_args.args[0]


def test_theme_color_command_resets_to_the_built_in(mocker, settings_path):
    from omni import ui
    settings_path.write_text(json.dumps({"themeColor": "#00b4d8", "mcpServers": {"d": {}}}))
    accent = mocker.patch("omni.ui.set_accent")
    cfg = AgentConfig(theme_color="#00b4d8")
    cli_mod._theme_color_command(cfg, None, "reset")
    accent.assert_called_once_with(ui.DEFAULT_ACCENT)
    assert cfg.theme_color == ""
    data = json.loads(settings_path.read_text())
    assert "themeColor" not in data and data["mcpServers"] == {"d": {}}


def test_theme_color_command_with_no_argument_reports(mocker, settings_path):
    settings_path.write_text(json.dumps({"themeColor": "#00b4d8"}))
    echoed = mocker.patch.object(cli_mod, "_echo")
    cli_mod._theme_color_command(AgentConfig(theme_color="#00b4d8"), None, "")
    said = echoed.call_args.args[0]
    assert "#00b4d8" in said and "saved" in said


def test_theme_color_applies_even_when_the_settings_file_cannot_be_written(mocker):
    """The colour still works for this session; refusing outright would be
    worse than saying the preference didn't stick."""
    accent = mocker.patch("omni.ui.set_accent")
    mocker.patch.object(cli_mod, "save_setting", side_effect=OSError("read-only"))
    echoed = mocker.patch.object(cli_mod, "_echo")
    cfg = AgentConfig()
    cli_mod._theme_color_command(cfg, None, "#00b4d8")
    accent.assert_called_once_with("#00b4d8")
    assert cfg.theme_color == "#00b4d8"
    assert "could not save" in echoed.call_args.args[0]


def test_theme_color_is_a_listed_command():
    assert "/theme-color" in cli_mod._STATIC_COMMANDS


# ---------------- where the model comes from ----------------

def test_the_model_is_discovered_from_the_server_when_nothing_names_one(mocker, tmp_path):
    """No model name is compiled in: the server knows which models exist and
    any built-in default would be the wrong one on most installs."""
    discover = mocker.patch.object(cli_mod, "_discover_model", return_value="server-model")
    captured = captured_cfg(mocker)
    r = invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].model == "server-model"
    assert "server-model" in r.output
    assert discover.call_count == 1


def test_a_saved_model_is_not_second_guessed_by_the_server(mocker, tmp_path, settings_path):
    settings_path.write_text(json.dumps({"model": "saved-model"}))
    discover = mocker.patch.object(cli_mod, "_discover_model")
    captured = captured_cfg(mocker)
    invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].model == "saved-model"
    discover.assert_not_called()


def test_the_flag_beats_everything(mocker, tmp_path, settings_path, monkeypatch):
    settings_path.write_text(json.dumps({"model": "saved-model"}))
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "env-model")
    discover = mocker.patch.object(cli_mod, "_discover_model")
    captured = captured_cfg(mocker)
    invoke("t", "--model", "flag-model", "--db-path", str(tmp_path / "d.db"),
           "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].model == "flag-model"
    discover.assert_not_called()


def test_the_env_var_names_a_model_when_nothing_is_saved(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "env-model")
    discover = mocker.patch.object(cli_mod, "_discover_model")
    captured = captured_cfg(mocker)
    invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].model == "env-model"
    discover.assert_not_called()


def test_a_saved_model_beats_the_env_var(mocker, tmp_path, settings_path, monkeypatch):
    """$DEFAULT_LLM_MODEL is a default — what to use when nothing else says —
    not an override of what you deliberately saved."""
    settings_path.write_text(json.dumps({"model": "saved-model"}))
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "env-model")
    captured = captured_cfg(mocker)
    invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].model == "saved-model"


def test_no_model_anywhere_refuses_to_start(mocker, tmp_path):
    """Better than a run that gets as far as its first call and fails inside
    the server, where the fix is much harder to read off."""
    mocker.patch.object(cli_mod, "_discover_model", return_value="")
    r = invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1
    assert "no model set" in r.output and "--model" in r.output


def test_discovery_takes_the_first_model_and_swallows_a_dead_server(mocker, no_model_discovery):
    """The real lookup, not the stand-in every other test runs with."""
    discover = no_model_discovery.real
    mocker.patch.object(cli_mod, "list_models", mocker.AsyncMock(return_value=["a", "b"]))
    assert discover("", "") == "a"
    mocker.patch.object(cli_mod, "list_models", mocker.AsyncMock(return_value=[]))
    assert discover("", "") == ""
    mocker.patch.object(cli_mod, "list_models",
                        mocker.AsyncMock(side_effect=LLMError("nothing listening")))
    assert discover("", "") == ""


def test_config_says_when_a_value_came_from_the_environment(mocker, monkeypatch, settings_path):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "env-model")
    echoed = mocker.patch.object(cli_mod, "_echo")
    cli_mod._setting_command(AgentConfig(model="env-model"), None,
                             cli_mod.SETTINGS_BY_NAME["model"], "")
    assert "[$DEFAULT_LLM_MODEL]" in echoed.call_args.args[0]


# ---------------- /system-prompt ----------------

def test_system_prompt_command_saves_and_applies(settings_path):
    cfg = AgentConfig()
    cli_mod._system_prompt_command(cfg, "Be terse.")
    assert cfg.system_prompt == "Be terse."
    assert json.loads(settings_path.read_text())["systemPrompt"] == "Be terse."


def test_system_prompt_command_reads_a_file(tmp_path, settings_path):
    """A prompt is usually more than one line and the REPL reads one line, so
    a file is how a real one gets set."""
    source = tmp_path / "prompt.md"
    source.write_text("Line one.\nLine two.\n")
    cfg = AgentConfig()
    cli_mod._system_prompt_command(cfg, f"file {source}")
    assert cfg.system_prompt == "Line one.\nLine two.\n"
    assert json.loads(settings_path.read_text())["systemPrompt"].startswith("Line one.")


def test_system_prompt_command_refuses_an_empty_file(mocker, tmp_path, settings_path):
    """Saving it would clear the prompt rather than replace it, which is not
    what "set it from this file" asked for."""
    source = tmp_path / "blank.md"
    source.write_text("\n \n")
    echoed = mocker.patch.object(cli_mod, "_echo")
    cfg = AgentConfig()
    cli_mod._system_prompt_command(cfg, f"file {source}")
    assert cfg.system_prompt == "" and not settings_path.exists()
    assert echoed.call_args.kwargs.get("err") is True


def test_system_prompt_command_reports_a_missing_file(mocker, tmp_path, settings_path):
    echoed = mocker.patch.object(cli_mod, "_echo")
    cli_mod._system_prompt_command(AgentConfig(), f"file {tmp_path / 'nope.md'}")
    assert not settings_path.exists()
    assert echoed.call_args.kwargs.get("err") is True


def test_system_prompt_command_resets_to_the_built_in(settings_path):
    settings_path.write_text(json.dumps({"systemPrompt": "Be terse.", "mcpServers": {"d": {}}}))
    cfg = AgentConfig(system_prompt="Be terse.")
    cli_mod._system_prompt_command(cfg, "reset")
    assert cfg.system_prompt == ""
    data = json.loads(settings_path.read_text())
    assert "systemPrompt" not in data and data["mcpServers"] == {"d": {}}


def test_system_prompt_command_with_no_argument_reports(mocker, settings_path):
    echoed = mocker.patch.object(cli_mod, "_echo")
    cli_mod._system_prompt_command(AgentConfig(), "")
    assert "built-in" in echoed.call_args.args[0]
    settings_path.write_text(json.dumps({"systemPrompt": "Be terse."}))
    cli_mod._system_prompt_command(AgentConfig(system_prompt="Be terse."), "")
    said = echoed.call_args.args[0]
    assert "Be terse." in said and "saved" in said


def test_system_prompt_applies_even_when_the_settings_file_cannot_be_written(mocker):
    mocker.patch.object(cli_mod, "save_setting", side_effect=OSError("read-only"))
    echoed = mocker.patch.object(cli_mod, "_echo")
    cfg = AgentConfig()
    cli_mod._system_prompt_command(cfg, "Be terse.")
    assert cfg.system_prompt == "Be terse."
    assert "could not save" in echoed.call_args.args[0]


def test_a_saved_system_prompt_is_used_when_the_flag_is_absent(mocker, tmp_path, settings_path):
    settings_path.write_text(json.dumps({"systemPrompt": "Be terse."}))
    captured = captured_cfg(mocker)
    invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].system_prompt == "Be terse."


def test_the_flag_overrides_the_saved_system_prompt_without_replacing_it(mocker, tmp_path,
                                                                        settings_path):
    settings_path.write_text(json.dumps({"systemPrompt": "Be terse."}))
    captured = captured_cfg(mocker)
    invoke("t", "--system-prompt", "Be verbose.", "--db-path", str(tmp_path / "d.db"),
           "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].system_prompt == "Be verbose."
    assert json.loads(settings_path.read_text())["systemPrompt"] == "Be terse."


# ---------------- /context-window-budget ----------------

def test_context_window_budget_command_saves_and_applies(settings_path):
    cfg = AgentConfig()
    cli_mod._context_window_budget_command(cfg, "50_000")
    assert cfg.context_window_budget == 50_000
    assert json.loads(settings_path.read_text())["contextWindowBudget"] == 50_000


def test_context_window_budget_command_refuses_junk_and_saves_nothing(mocker, settings_path):
    echoed = mocker.patch.object(cli_mod, "_echo")
    cfg = AgentConfig()
    cli_mod._context_window_budget_command(cfg, "lots")
    assert cfg.context_window_budget == AgentConfig.context_window_budget
    assert not settings_path.exists()
    assert echoed.call_args.kwargs.get("err") is True


def test_context_window_budget_command_refuses_a_budget_under_the_floor(mocker, settings_path):
    """Under the floor the loop compacts on every step — a summarization call
    per turn that saves nothing."""
    echoed = mocker.patch.object(cli_mod, "_echo")
    cfg = AgentConfig()
    cli_mod._context_window_budget_command(cfg, "12")
    assert cfg.context_window_budget == AgentConfig.context_window_budget
    assert not settings_path.exists()
    assert echoed.call_args.kwargs.get("err") is True


def test_context_window_budget_command_resets_to_the_default(settings_path):
    settings_path.write_text(json.dumps({"contextWindowBudget": 50_000, "mcpServers": {"d": {}}}))
    cfg = AgentConfig(context_window_budget=50_000)
    cli_mod._context_window_budget_command(cfg, "reset")
    assert cfg.context_window_budget == AgentConfig.context_window_budget
    data = json.loads(settings_path.read_text())
    assert "contextWindowBudget" not in data and data["mcpServers"] == {"d": {}}


def test_context_window_budget_command_with_no_argument_reports(mocker, settings_path):
    echoed = mocker.patch.object(cli_mod, "_echo")
    cli_mod._context_window_budget_command(AgentConfig(), "")
    said = echoed.call_args.args[0]
    assert f"{AgentConfig.context_window_budget:,}" in said and "[saved]" not in said
    settings_path.write_text(json.dumps({"contextWindowBudget": 80_000}))
    cli_mod._context_window_budget_command(AgentConfig(context_window_budget=80_000), "")
    assert "[saved]" in echoed.call_args.args[0]


def test_context_window_budget_applies_even_when_the_settings_file_cannot_be_written(mocker):
    mocker.patch.object(cli_mod, "save_setting", side_effect=OSError("read-only"))
    echoed = mocker.patch.object(cli_mod, "_echo")
    cfg = AgentConfig()
    cli_mod._context_window_budget_command(cfg, "50000")
    assert cfg.context_window_budget == 50_000
    assert "could not save" in echoed.call_args.args[0]


def test_a_saved_context_window_budget_is_used_when_the_flag_is_absent(mocker, tmp_path,
                                                                    settings_path):
    settings_path.write_text(json.dumps({"contextWindowBudget": 50_000}))
    captured = captured_cfg(mocker)
    invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].context_window_budget == 50_000


def test_the_flag_overrides_the_saved_budget_without_replacing_it(mocker, tmp_path, settings_path):
    settings_path.write_text(json.dumps({"contextWindowBudget": 50_000}))
    captured = captured_cfg(mocker)
    invoke("t", "--context-window-budget", "90000", "--db-path", str(tmp_path / "d.db"),
           "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].context_window_budget == 90_000
    assert json.loads(settings_path.read_text())["contextWindowBudget"] == 50_000


def test_a_budget_flag_under_the_floor_exits_nonzero(tmp_path):
    r = invoke("t", "--context-window-budget", "12", "--db-path", str(tmp_path / "d.db"),
               "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1 and "compacts on every step" in r.output


def test_every_setting_is_a_listed_command():
    """The completion menu is generated from the registry, so a new setting
    is a new command without a second list to keep in step."""
    for setting in cli_mod.SETTINGS:
        assert f"/{setting.name}" in cli_mod._STATIC_COMMANDS
    assert "/config" in cli_mod._STATIC_COMMANDS


# ---------------- the rest of the registry ----------------

def test_any_setting_saves_and_applies(settings_path):
    cfg = AgentConfig()
    cli_mod._setting_command(cfg, None, cli_mod.SETTINGS_BY_NAME["max-steps"], "7")
    cli_mod._setting_command(cfg, None, cli_mod.SETTINGS_BY_NAME["parse-intent"], "off")
    assert cfg.max_steps == 7 and cfg.parse_intent is False
    saved = json.loads(settings_path.read_text())
    assert saved == {"maxSteps": 7, "parseIntent": False}


def test_a_setting_that_only_takes_hold_later_says_so(mocker, settings_path):
    """The tool-side knobs are handed to the MCP server process as
    environment when it connects, so a change now is a change next run."""
    echoed = mocker.patch.object(cli_mod, "_echo")
    cli_mod._setting_command(AgentConfig(), None, cli_mod.SETTINGS_BY_NAME["shell-timeout"], "60")
    assert "next time omni starts" in echoed.call_args.args[0]


def test_switching_model_through_a_setting_announces_it(mocker, settings_path):
    announced = mocker.patch.object(cli_mod, "_announce_model")
    cfg = AgentConfig()
    cli_mod._setting_command(cfg, None, cli_mod.SETTINGS_BY_NAME["model"], "other-model")
    announced.assert_called_once_with("other-model", None)
    assert json.loads(settings_path.read_text())["model"] == "other-model"


def test_config_lists_every_setting_and_marks_the_saved_ones(mocker, settings_path):
    settings_path.write_text(json.dumps({"maxSteps": 7}))
    echoed = mocker.patch.object(cli_mod, "_echo")
    cli_mod._config_command(AgentConfig(max_steps=7), None, "")
    listing = echoed.call_args.args[0]
    for setting in cli_mod.SETTINGS:
        assert f"/{setting.name}" in listing
    assert "[saved]" in listing


def test_config_reset_clears_every_saved_setting_but_not_the_mcp_servers(settings_path):
    settings_path.write_text(json.dumps({"maxSteps": 7, "parseIntent": False,
                                         "mcpServers": {"docs": {"command": "node"}}}))
    cfg = AgentConfig(max_steps=7, parse_intent=False)
    cli_mod._config_command(cfg, None, "reset")
    assert cfg.max_steps == AgentConfig.max_steps and cfg.parse_intent is True
    assert json.loads(settings_path.read_text()) == {"mcpServers": {"docs": {"command": "node"}}}


def test_config_reset_with_nothing_saved_says_so(mocker, settings_path):
    echoed = mocker.patch.object(cli_mod, "_echo")
    cli_mod._config_command(AgentConfig(), None, "reset")
    assert "already at its default" in echoed.call_args.args[0]


def test_config_reports_one_setting_by_name(mocker, settings_path):
    echoed = mocker.patch.object(cli_mod, "_echo")
    cli_mod._config_command(AgentConfig(), None, "max_steps")
    assert echoed.call_args.args[0].startswith("max-steps:")


def test_a_saved_setting_is_used_when_its_flag_is_absent(mocker, tmp_path, settings_path):
    settings_path.write_text(json.dumps({"maxSteps": 7, "model": "saved-model",
                                         "parseIntent": False, "shellTimeoutS": 60}))
    captured = captured_cfg(mocker)
    invoke("t", "--db-path", str(tmp_path / "d.db"), "--log-path", str(tmp_path / "l.log"))
    cfg = captured["cfg"]
    assert cfg.max_steps == 7 and cfg.model == "saved-model"
    assert cfg.parse_intent is False and cfg.shell_timeout_s == 60


def test_a_flag_overrides_the_saved_setting_without_replacing_it(mocker, tmp_path, settings_path):
    settings_path.write_text(json.dumps({"maxSteps": 7, "model": "saved-model"}))
    captured = captured_cfg(mocker)
    invoke("t", "--max-steps", "9", "--db-path", str(tmp_path / "d.db"),
           "--log-path", str(tmp_path / "l.log"))
    assert captured["cfg"].max_steps == 9 and captured["cfg"].model == "saved-model"
    assert json.loads(settings_path.read_text())["maxSteps"] == 7


def test_a_flag_value_the_setting_would_refuse_exits_nonzero(tmp_path):
    """The flags are validated the same way the slash commands are, so
    --max-steps 0 is refused at the door instead of producing a run that
    ends before its first step."""
    r = invoke("t", "--max-steps", "0", "--db-path", str(tmp_path / "d.db"),
               "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1 and "--max-steps" in r.output


def test_run_value_error_exits_nonzero(mocker, tmp_path):
    mocker.patch.object(cli_mod.CodingAgent, "run",
                        mocker.AsyncMock(side_effect=ValueError("no session found")))
    r = invoke("t", "--db-path", str(tmp_path / "s.db"), "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1 and "no session found" in r.output


def test_unreachable_llm_server_is_one_line_not_a_traceback(mocker, tmp_path):
    """What _call_model raises once its retries are spent — the single most
    common failure mode, and previously an uncaught stack trace."""
    mocker.patch.object(cli_mod.CodingAgent, "run", mocker.AsyncMock(
        side_effect=RuntimeError("Model call failed after 3 attempts: Could not reach the LLM server")))
    r = invoke("t", "--db-path", str(tmp_path / "s.db"), "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1
    assert "Could not reach the LLM server" in r.output
    assert "Traceback" not in r.output


def test_llm_error_from_a_run_exits_nonzero(mocker, tmp_path):
    mocker.patch.object(cli_mod.CodingAgent, "run",
                        mocker.AsyncMock(side_effect=LLMError("bad gateway")))
    r = invoke("t", "--db-path", str(tmp_path / "s.db"), "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 1 and "bad gateway" in r.output


def test_no_task_enters_interactive(no_interactive, tmp_path):
    r = invoke("--db-path", str(tmp_path / "s.db"), "--log-path", str(tmp_path / "l.log"))
    assert r.exit_code == 0
    no_interactive.assert_awaited_once()


def test_help_lists_the_key_flags():
    out = invoke("--help").output
    for flag in ("--project-root", "--model", "--llm-host", "--llm-timeout",
                 "--auto-approve", "--safe-tool", "--resume", "--add-mcp-server",
                 "--mcp-log-path", "--system-prompt", "--system-prompt-file",
                 "--mcp-connect-timeout", "--theme-color"):
        assert flag in out


# ---------------- MCP registry commands ----------------

def test_add_mcp_server_writes_to_the_settings_file(settings_path):
    r = invoke("--add-mcp-server", "weather=python -m weather_srv")
    assert r.exit_code == 0 and "Registered" in r.output
    data = json.loads(settings_path.read_text())
    assert data["mcpServers"]["weather"]["command"] == "python"
    assert data["mcpServers"]["weather"]["args"] == ["-m", "weather_srv"]


def test_add_mcp_server_with_defer_flag(settings_path):
    invoke("--add-mcp-server", "docs=node srv.js", "--defer")
    assert json.loads(settings_path.read_text())["mcpServers"]["docs"]["defer"] is True


def test_add_mcp_server_with_bearer_keeps_env_reference(settings_path):
    invoke("--add-mcp-server", "docs=https://h/mcp/sse,bearer=$DOCS_TOKEN")
    entry = json.loads(settings_path.read_text())["mcpServers"]["docs"]
    assert entry["headers"] == {"Authorization": "Bearer $DOCS_TOKEN"}


def test_add_mcp_server_preserves_other_settings_keys(settings_path):
    settings_path.write_text(json.dumps({"theme": "dark", "mcpServers": {}}))
    invoke("--add-mcp-server", "w=python -m w")
    data = json.loads(settings_path.read_text())
    assert data["theme"] == "dark" and "w" in data["mcpServers"]


def test_add_mcp_server_rejects_malformed_spec(settings_path):
    r = invoke("--add-mcp-server", "no-equals-sign")
    assert r.exit_code == 1 and "Error" in r.output
    assert not settings_path.exists()


def test_list_mcp_servers(settings_path):
    settings_path.write_text(json.dumps({"mcpServers": {
        "w": {"command": "python", "args": ["-m", "w"]},
        "d": {"url": "https://h/mcp", "defer": True},
    }}))
    out = invoke("--list-mcp-servers").output
    assert "w: python -m w" in out
    assert "d: https://h/mcp [defer]" in out


def test_list_mcp_servers_when_empty():
    assert "No registered MCP servers" in invoke("--list-mcp-servers").output


def test_remove_mcp_server(settings_path):
    settings_path.write_text(json.dumps({"mcpServers": {"w": {"command": "python"}}}))
    r = invoke("--remove-mcp-server", "w")
    assert r.exit_code == 0 and "Removed" in r.output
    assert json.loads(settings_path.read_text())["mcpServers"] == {}


def test_remove_unknown_mcp_server_errors(settings_path):
    settings_path.write_text(json.dumps({"mcpServers": {}}))
    r = invoke("--remove-mcp-server", "ghost")
    assert r.exit_code == 1 and "no registered MCP server" in r.output


# ---------------- session management commands ----------------

@pytest.fixture
def store(tmp_path):
    from omni.session_store import SessionStore
    return SessionStore(str(tmp_path / "s.db"))


def test_list_sessions(store, tmp_path):
    store.create_session("/p", "qwen", "my saved task", name="nm")
    out = invoke("--list-sessions", "--db-path", str(tmp_path / "s.db")).output
    assert "my saved task" in out and "nm" in out


def test_list_sessions_when_empty(tmp_path):
    out = invoke("--list-sessions", "--db-path", str(tmp_path / "empty.db")).output
    assert "No saved sessions" in out


def test_delete_session(store, tmp_path):
    sid = store.create_session("/p", "m", "t", name="doomed")
    r = invoke("--delete-session", "doomed", "--db-path", str(tmp_path / "s.db"))
    assert r.exit_code == 0 and "Deleted" in r.output
    assert not store.session_exists(sid)


def test_delete_unknown_session_errors(tmp_path):
    r = invoke("--delete-session", "ghost", "--db-path", str(tmp_path / "s.db"))
    assert r.exit_code == 1 and "no session found" in r.output.lower()


# ---------------- _handle_btw ----------------

async def test_handle_btw_answers_without_touching_history(cfg, mocker):
    chat = mocker.patch.object(cli_mod, "chat", mocker.AsyncMock(return_value={"content": "42"}))
    shown = mocker.patch("omni.ui.btw_answer")
    await cli_mod._handle_btw(cfg, "what is 6*7?")

    shown.assert_called_once_with("what is 6*7?", "42")
    sent = chat.await_args.kwargs["messages"]
    assert len(sent) == 2 and sent[1]["content"] == "what is 6*7?"   # stateless, no history


async def test_handle_btw_uses_configured_model_and_host(cfg, mocker):
    cfg.llm_host, cfg.llm_api_key, cfg.llm_timeout_s = "http://h", "k", 33.0
    chat = mocker.patch.object(cli_mod, "chat", mocker.AsyncMock(return_value={"content": "x"}))
    mocker.patch("omni.ui.btw_answer")
    await cli_mod._handle_btw(cfg, "q")
    kwargs = chat.await_args.kwargs
    assert kwargs["model"] == cfg.model and kwargs["base_url"] == "http://h"
    assert kwargs["api_key"] == "k" and kwargs["timeout"] == 33.0


async def test_handle_btw_reports_llm_errors_inline(cfg, mocker):
    mocker.patch.object(cli_mod, "chat", mocker.AsyncMock(side_effect=LLMError("server down")))
    shown = mocker.patch("omni.ui.btw_answer")
    await cli_mod._handle_btw(cfg, "q")
    assert "server down" in shown.call_args.args[1]


async def test_handle_btw_handles_empty_response(cfg, mocker):
    mocker.patch.object(cli_mod, "chat", mocker.AsyncMock(return_value={"content": None}))
    shown = mocker.patch("omni.ui.btw_answer")
    await cli_mod._handle_btw(cfg, "q")
    assert "(empty response)" in shown.call_args.args[1]


# ---------------- REPL helpers ----------------

def test_print_header_delegates_to_ui(mocker, cfg):
    """No model in the header: it would be stale after the first /model
    switch, and the frame's hint line carries the live one instead."""
    header = mocker.patch("omni.ui.header")
    cli_mod._print_header(cfg, "my-label")
    header.assert_called_once_with("my-label", cfg.project_root)


def test_model_switch_is_announced_without_a_header(mocker):
    switched = mocker.patch("omni.ui.model_switched")
    header = mocker.patch("omni.ui.header")
    cli_mod._announce_model("llama3.1:latest")
    switched.assert_called_once_with("llama3.1:latest")
    header.assert_not_called()


def test_print_sessions_empty_and_populated(mocker, capsys):
    table = mocker.patch("omni.ui.sessions_table")
    cli_mod._print_sessions([])
    assert "No saved sessions" in capsys.readouterr().out
    table.assert_not_called()

    rows = [{"id": "a", "name": None, "status": "done", "updated_at": "t", "model": "m", "task": "x"}]
    cli_mod._print_sessions(rows)
    table.assert_called_once_with(rows)


def test_print_resources_delegates_to_ui(mocker):
    table = mocker.patch("omni.ui.resources_table")
    cli_mod._print_resources({"x://1": {"server": "d", "template": False}})
    table.assert_called_once()


def test_print_server_tools_delegates_to_ui(mocker):
    table = mocker.patch("omni.ui.server_tools_table")
    tools = [{"name": "docs__search", "real_name": "search", "description": "d",
              "deferred": False, "revealed": False, "internal": False}]
    cli_mod._print_server_tools("docs", tools)
    table.assert_called_once_with("docs", tools)


def test_print_mcp_status_delegates_to_ui(mocker):
    status = mocker.patch("omni.ui.mcp_status")
    cli_mod._print_mcp_status([{"name": "built-in", "connected": True}])
    status.assert_called_once()


def _resume_agent(tmp_path):
    from omni.session_store import SessionStore
    return SimpleNamespace(store=SessionStore(str(tmp_path / "s.db")), call_log=[])


def test_show_resumed_history_replays_when_session_exists(mocker, tmp_path):
    agent = _resume_agent(tmp_path)
    sid = agent.store.create_session("/p", "m", "t")
    agent.store.append_message(sid, 0, {"role": "user", "content": "earlier"})
    replay = mocker.patch("omni.ui.replay_history", return_value=[])
    cli_mod._show_resumed_history(agent, sid)
    assert replay.call_args.args[0][0][0]["content"] == "earlier"


def test_show_resumed_history_seeds_the_call_log(mocker, tmp_path):
    """Replayed calls keep their numbers, so /expand <n> still reaches them
    and the next live call doesn't reuse a number already on screen."""
    agent = _resume_agent(tmp_path)
    sid = agent.store.create_session("/p", "m", "t")
    agent.store.append_message(sid, 0, {"role": "user", "content": "earlier"})
    mocker.patch("omni.ui.replay_history", return_value=[{"index": 1, "name": "read_file"}])
    cli_mod._show_resumed_history(agent, sid)
    assert [r["index"] for r in agent.call_log] == [1]


def test_show_resumed_history_silent_for_unknown_session(mocker, tmp_path):
    """Stays quiet so agent.run() can raise the proper error instead."""
    replay = mocker.patch("omni.ui.replay_history")
    cli_mod._show_resumed_history(_resume_agent(tmp_path), "ghost")
    replay.assert_not_called()


@pytest.mark.parametrize("args, expected", [
    (["--resume"], ["--resume", cli_mod.PICK_SESSION]),
    (["--resume", "abc"], ["--resume", "abc"]),
    (["--resume=abc"], ["--resume=abc"]),
    (["--resume", "-p", "/x"], ["--resume", cli_mod.PICK_SESSION, "-p", "/x"]),
    (["--resume", "--"], ["--resume", cli_mod.PICK_SESSION, "--"]),
    (["a task", "--resume"], ["a task", "--resume", cli_mod.PICK_SESSION]),
    (["--model", "m"], ["--model", "m"]),
])
def test_fill_bare_resume(args, expected):
    """--resume has to work both with an id and on its own; Typer can't
    express a value-optional option, so the value is filled in first."""
    assert cli_mod.fill_bare_resume(args) == expected


def test_bare_resume_opens_the_picker_and_resumes_what_it_returns(mocker, tmp_path):
    picked = mocker.patch.object(cli_mod, "_pick_session", return_value="chosen-id")
    interactive = mocker.patch.object(cli_mod, "_interactive", new=mocker.AsyncMock())
    result = CliRunner().invoke(app, ["--resume", "--model", "m", "-p", str(tmp_path)])
    assert result.exit_code == 0, result.output
    picked.assert_called_once()
    assert interactive.await_args.args[1] == "chosen-id"


def test_a_dismissed_picker_starts_nothing(mocker, tmp_path):
    """Escaping the list means "never mind", not "start a fresh session"."""
    mocker.patch.object(cli_mod, "_pick_session", return_value=None)
    interactive = mocker.patch.object(cli_mod, "_interactive", new=mocker.AsyncMock())
    result = CliRunner().invoke(app, ["--resume", "--model", "m", "-p", str(tmp_path)])
    assert result.exit_code == 0
    interactive.assert_not_awaited()


def test_resume_with_an_id_never_opens_the_picker(mocker, tmp_path):
    picked = mocker.patch.object(cli_mod, "_pick_session")
    mocker.patch.object(cli_mod, "_interactive", new=mocker.AsyncMock())
    CliRunner().invoke(app, ["--resume", "abc123", "--model", "m", "-p", str(tmp_path)])
    picked.assert_not_called()


def test_the_picker_is_handed_this_projects_sessions_and_branch(mocker, cfg):
    from omni.session_store import SessionStore
    store = SessionStore(cfg.db_path)
    store.create_session(cfg.project_root, "m", "earlier task", name="earlier")
    mocker.patch.object(cli_mod, "current_branch", return_value="feature/x")
    picker = mocker.patch("omni.session_picker.SessionPicker")
    picker.return_value.run = mocker.AsyncMock(return_value="picked")

    assert cli_mod._pick_session(cfg) == "picked"
    sessions, project_root, branch = picker.call_args.args
    assert [s["name"] for s in sessions] == ["earlier"]
    assert project_root == cfg.project_root and branch == "feature/x"


def test_picking_with_no_saved_sessions_says_so(mocker, cfg, capsys):
    picker = mocker.patch("omni.session_picker.SessionPicker")
    assert cli_mod._pick_session(cfg) is None
    picker.assert_not_called()
    assert "No saved sessions" in capsys.readouterr().err


async def test_restart_mcp_server_wraps_the_client_call(mocker):
    client = mocker.AsyncMock()
    client.restart_server.return_value = {"name": "docs", "connected": True, "tool_count": 3}
    mocker.patch("omni.ui.thinking")
    out = await cli_mod._restart_mcp_server(client, "docs")
    assert out["connected"] is True
    client.restart_server.assert_awaited_once_with("docs")


# ---------------- static command registry ----------------

@pytest.mark.parametrize("command", [
    "/exit", "/quit", "/sessions", "/delete ", "/compact",
    "/btw ", "/model", "/mcp", "/mcp restart ", "/resources",
])
def test_static_commands_are_registered_with_descriptions(command):
    assert command in cli_mod._STATIC_COMMANDS
    assert cli_mod._STATIC_COMMANDS[command].strip()
