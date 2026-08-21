"""mcp_client's pure/helper layer: spec parsing, settings file IO, env-var
expansion, schema shaping, and the benign-shutdown-race classifier."""

import json

import pytest
from anyio import BrokenResourceError

from omni import mcp_client as mc
from omni.mcp_client import (
    _expand_env_values, _is_benign_shutdown_race, _mcp_schema_to_tool_schema,
    _resource_tool_schemas, _search_tools_schema, _split_command,
    default_mcp_config_path, load_mcp_config, parse_mcp_server_specs, save_mcp_config,
)


# ---------------- _is_benign_shutdown_race ----------------

def test_benign_race_recognizes_broken_resource():
    assert _is_benign_shutdown_race(BrokenResourceError()) is True


def test_benign_race_rejects_real_errors():
    assert _is_benign_shutdown_race(RuntimeError("real")) is False


def test_benign_race_unwraps_nested_exception_groups():
    inner = BaseExceptionGroup("g", [BrokenResourceError()])
    assert _is_benign_shutdown_race(BaseExceptionGroup("outer", [inner])) is True


def test_benign_race_group_with_any_real_error_is_not_benign():
    group = BaseExceptionGroup("g", [BrokenResourceError(), ValueError("real")])
    assert _is_benign_shutdown_race(group) is False


# ---------------- _split_command ----------------

def test_split_command_basic():
    assert _split_command("python -m srv --port 4000") == ["python", "-m", "srv", "--port", "4000"]


def test_split_command_preserves_windows_backslashes():
    """POSIX-mode shlex eats backslashes, mangling C:\\Users\\... paths."""
    out = _split_command(r"python C:\Users\me\srv.py")
    assert out == ["python", r"C:\Users\me\srv.py"]


def test_split_command_strips_matched_quotes():
    assert _split_command('node "my server.js"') == ["node", "my server.js"]
    assert _split_command("node 'my server.js'") == ["node", "my server.js"]


def test_split_command_empty():
    assert _split_command("") == []


# ---------------- parse_mcp_server_specs ----------------

def test_parse_stdio_spec():
    out = parse_mcp_server_specs(["weather=python -m weather_srv"])
    assert out == {"weather": {"command": "python", "args": ["-m", "weather_srv"], "defer": False}}


def test_parse_remote_defaults_to_sse():
    out = parse_mcp_server_specs(["docs=https://x/mcp/sse"])["docs"]
    assert out["url"] == "https://x/mcp/sse" and out["transport"] == "sse"


def test_parse_remote_explicit_streamable_http():
    out = parse_mcp_server_specs(["docs=https://x/mcp,streamable_http"])["docs"]
    assert out["transport"] == "streamable_http"


def test_parse_http_scheme_also_treated_as_remote():
    assert "url" in parse_mcp_server_specs(["d=http://x/mcp"])["d"]


def test_parse_defer_suffix():
    assert parse_mcp_server_specs(["d=python -m s,defer"])["d"]["defer"] is True
    assert parse_mcp_server_specs(["d=https://x/mcp,streamable_http,defer"])["d"]["defer"] is True


def test_parse_bearer_becomes_authorization_header():
    out = parse_mcp_server_specs(["d=https://x/mcp/sse,bearer=sk-abc"])["d"]
    assert out["headers"] == {"Authorization": "Bearer sk-abc"}


def test_parse_bearer_keeps_env_reference_unexpanded():
    """--add-mcp-server persists this spec, so the reference (not the
    resolved secret) is what must be stored."""
    out = parse_mcp_server_specs(["d=https://x/mcp/sse,bearer=$TOKEN"])["d"]
    assert out["headers"] == {"Authorization": "Bearer $TOKEN"}


@pytest.mark.parametrize("spec", [
    "d=https://x/mcp,streamable_http,bearer=$T,defer",
    "d=https://x/mcp,streamable_http,defer,bearer=$T",
])
def test_parse_bearer_and_defer_in_either_order(spec):
    out = parse_mcp_server_specs([spec])["d"]
    assert out["defer"] is True and out["transport"] == "streamable_http"
    assert out["headers"] == {"Authorization": "Bearer $T"}


def test_parse_bearer_rejected_for_stdio():
    with pytest.raises(ValueError, match="only valid for remote"):
        parse_mcp_server_specs(["d=python -m s,bearer=tok"])


def test_parse_empty_bearer_rejected():
    with pytest.raises(ValueError, match="Empty"):
        parse_mcp_server_specs(["d=https://x/mcp,bearer="])


def test_parse_multiple_specs_and_empty_input():
    out = parse_mcp_server_specs(["a=python -m a", "b=https://x/mcp/sse"])
    assert set(out) == {"a", "b"}
    assert parse_mcp_server_specs([]) == {}
    assert parse_mcp_server_specs(None) == {}


@pytest.mark.parametrize("bad", ["noequals", "=python -m s", "   =x"])
def test_parse_rejects_malformed_specs(bad):
    with pytest.raises(ValueError):
        parse_mcp_server_specs([bad])


def test_parse_rejects_unknown_transport():
    with pytest.raises(ValueError, match="unknown transport|transport"):
        parse_mcp_server_specs(["d=https://x/mcp,carrier-pigeon"])


def test_parse_rejects_empty_command():
    with pytest.raises(ValueError):
        parse_mcp_server_specs(["d="])


# ---------------- settings file: path, migration, save/load ----------------

def test_default_path_is_settings_json_under_dot_omni_coder(mocker, tmp_path):
    mocker.patch("os.path.expanduser", lambda p: p.replace("~", str(tmp_path)))
    (tmp_path / ".omni-coder").mkdir()
    assert default_mcp_config_path() == str(tmp_path / ".omni-coder" / "omni-coder-settings.json")


def test_default_path_migrates_legacy_mcp_json(mocker, tmp_path):
    mocker.patch("os.path.expanduser", lambda p: p.replace("~", str(tmp_path)))
    d = tmp_path / ".omni-coder"
    d.mkdir()
    legacy = d / "mcp.json"
    legacy.write_text(json.dumps({"mcpServers": {"w": {"command": "python"}}}))

    path = default_mcp_config_path()
    assert path.endswith("omni-coder-settings.json")
    assert not legacy.exists()                      # moved, not copied
    assert load_mcp_config(path) == {"w": {"command": "python"}}


def test_default_path_prefers_new_file_when_both_exist(mocker, tmp_path):
    mocker.patch("os.path.expanduser", lambda p: p.replace("~", str(tmp_path)))
    d = tmp_path / ".omni-coder"
    d.mkdir()
    (d / "mcp.json").write_text(json.dumps({"mcpServers": {"old": {"command": "x"}}}))
    (d / "omni-coder-settings.json").write_text(json.dumps({"mcpServers": {"new": {"command": "y"}}}))
    assert load_mcp_config(default_mcp_config_path()) == {"new": {"command": "y"}}


def test_save_mcp_config_creates_dirs_and_roundtrips(tmp_path):
    path = str(tmp_path / "nested" / "settings.json")
    save_mcp_config(path, {"w": {"command": "python", "args": ["-m", "w"]}})
    assert load_mcp_config(path) == {"w": {"command": "python", "args": ["-m", "w"]}}


def test_save_mcp_config_preserves_unrelated_top_level_keys(tmp_path):
    """It's a general settings file — registering a server must not clobber
    anything else in it."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"theme": "dark", "llmHost": "http://h", "mcpServers": {}}))
    save_mcp_config(str(path), {"w": {"command": "python"}})
    data = json.loads(path.read_text())
    assert data["theme"] == "dark" and data["llmHost"] == "http://h"
    assert set(data["mcpServers"]) == {"w"}


def test_save_mcp_config_rewrites_corrupt_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json at all")
    save_mcp_config(str(path), {"w": {"command": "python"}})
    assert load_mcp_config(str(path)) == {"w": {"command": "python"}}


def test_save_mcp_config_replaces_non_dict_json(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("[1, 2, 3]")
    save_mcp_config(str(path), {"w": {"command": "python"}})
    assert load_mcp_config(str(path)) == {"w": {"command": "python"}}


def test_load_mcp_config_missing_key_is_empty(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"other": 1}))
    assert load_mcp_config(str(path)) == {}


@pytest.mark.parametrize("content", ["{bad json", ""])
def test_load_mcp_config_rejects_unparsable(tmp_path, content):
    path = tmp_path / "s.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="Could not read"):
        load_mcp_config(str(path))


def test_load_mcp_config_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="Could not read"):
        load_mcp_config(str(tmp_path / "nope.json"))


def test_load_mcp_config_validates_entries(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"mcpServers": {"x": {"env": {}}}}))
    with pytest.raises(ValueError, match="must have either"):
        load_mcp_config(str(path))

    path.write_text(json.dumps({"mcpServers": {"x": {"url": "http://h", "transport": "smoke"}}}))
    with pytest.raises(ValueError, match="unknown transport"):
        load_mcp_config(str(path))


def test_load_mcp_config_accepts_valid_remote_and_stdio(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"mcpServers": {
        "a": {"command": "python", "args": ["-m", "a"], "env": {"K": "v"}},
        "b": {"url": "https://h/mcp", "transport": "streamable_http",
              "headers": {"Authorization": "Bearer x"}, "defer": True},
    }}))
    out = load_mcp_config(str(path))
    assert out["b"]["defer"] is True and out["a"]["env"] == {"K": "v"}


# ---------------- _expand_env_values ----------------

def test_expand_env_resolves_both_syntaxes(monkeypatch):
    monkeypatch.setenv("TOK", "secret")
    assert _expand_env_values("s", {"A": "Bearer $TOK"}, "header") == {"A": "Bearer secret"}
    assert _expand_env_values("s", {"A": "Bearer ${TOK}"}, "header") == {"A": "Bearer secret"}


def test_expand_env_leaves_plain_values_alone(monkeypatch):
    assert _expand_env_values("s", {"A": "literal"}, "header") == {"A": "literal"}


def test_expand_env_passes_through_empty_and_none():
    assert _expand_env_values("s", None, "header") is None
    assert _expand_env_values("s", {}, "header") == {}


def test_expand_env_non_string_values_untouched():
    assert _expand_env_values("s", {"n": 5, "b": True}, "env var") == {"n": 5, "b": True}


def test_expand_env_unset_variable_raises_naming_it(monkeypatch):
    monkeypatch.delenv("NOPE_MISSING", raising=False)
    with pytest.raises(ValueError, match="NOPE_MISSING"):
        _expand_env_values("docs", {"Authorization": "Bearer $NOPE_MISSING"}, "header")


def test_expand_env_error_message_includes_server_and_kind(monkeypatch):
    monkeypatch.delenv("GONE", raising=False)
    with pytest.raises(ValueError, match=r"'docs'.*env var.*'K'"):
        _expand_env_values("docs", {"K": "$GONE"}, "env var")


def test_expand_env_reports_every_missing_variable(monkeypatch):
    for v in ("M1", "M2"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(ValueError) as e:
        _expand_env_values("s", {"K": "$M1 and $M2"}, "header")
    assert "M1" in str(e.value) and "M2" in str(e.value)


# ---------------- schema shaping ----------------

def test_mcp_schema_to_tool_schema(mocker):
    tool = mocker.Mock(description="Reads a file", inputSchema={"type": "object", "properties": {"p": {}}})
    out = _mcp_schema_to_tool_schema(tool, "srv__read")
    assert out["type"] == "function"
    assert out["function"]["name"] == "srv__read"
    assert out["function"]["description"] == "Reads a file"
    assert out["function"]["parameters"]["properties"] == {"p": {}}


def test_mcp_schema_defaults_for_missing_description_and_schema(mocker):
    tool = mocker.Mock(description=None, inputSchema=None)
    out = _mcp_schema_to_tool_schema(tool, "n")["function"]
    assert out["description"] == ""
    assert out["parameters"] == {"type": "object", "properties": {}}


def test_search_tools_schema_shape():
    fn = _search_tools_schema()["function"]
    assert fn["name"] == "search_tools"
    assert fn["parameters"]["required"] == ["query"]


def test_resource_tool_schemas_expose_both_tools_and_inline_uris():
    resources = {
        "file:///a.md": {"template": False},
        "file:///b.json": {"template": False},
        "file:///logs/{d}.log": {"template": True},
    }
    schemas = _resource_tool_schemas(resources)
    names = [s["function"]["name"] for s in schemas]
    assert names == ["list_resources", "read_resource"]

    read_desc = schemas[1]["function"]["description"]
    assert "file:///a.md" in read_desc and "file:///b.json" in read_desc
    assert "{d}" not in read_desc          # templates aren't offered as readable
    assert schemas[1]["function"]["parameters"]["required"] == ["uri"]
    assert schemas[0]["function"]["parameters"]["properties"] == {}


def test_resource_tool_schemas_caps_the_inlined_catalog():
    resources = {f"file:///f{n}.md": {"template": False} for n in range(30)}
    desc = _resource_tool_schemas(resources)[1]["function"]["description"]
    assert "10 more" in desc and "list_resources" in desc


def test_resource_tool_schemas_with_only_templates_says_none():
    desc = _resource_tool_schemas({"x://{a}": {"template": True}})[1]["function"]["description"]
    assert "(none)" in desc
