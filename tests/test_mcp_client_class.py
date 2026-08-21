"""MCPToolClient behaviour with mocked ClientSessions.

Sessions are AsyncMocks standing in for connected MCP servers, so tool
routing, deferred loading, search ranking, prompts, resources, and restart
bookkeeping are all exercised without spawning subprocesses.
"""

import asyncio

import pytest

from omni import mcp_client as mc
from omni.mcp_client import _BUILTIN, EmbeddingUnavailable, MCPToolClient


def tool(name, description="d", schema=None):
    t = type("T", (), {})()
    t.name = name
    t.description = description
    t.inputSchema = schema or {"type": "object", "properties": {}}
    return t


def fake_session(mocker, tools=(), prompts=(), resources=(), templates=()):
    s = mocker.AsyncMock()
    s.list_tools.return_value = mocker.Mock(tools=list(tools))
    s.list_prompts.return_value = mocker.Mock(prompts=list(prompts))
    s.list_resources.return_value = mocker.Mock(resources=list(resources))
    s.list_resource_templates.return_value = mocker.Mock(resourceTemplates=list(templates))
    return s


@pytest.fixture
def client(tmp_path):
    """A client with no real connections — sessions get injected per test."""
    c = MCPToolClient(str(tmp_path), mcp_log_path=str(tmp_path / "mcp.log"))
    return c


# ---------------- list_llm_tools: namespacing & filtering ----------------

async def test_builtin_tools_keep_plain_names(client, mocker):
    client._sessions[_BUILTIN] = fake_session(mocker, [tool("read_file"), tool("write_file")])
    names = [s["function"]["name"] for s in await client.list_llm_tools()]
    assert names == ["read_file", "write_file"]


async def test_builtin_underscore_tools_are_hidden_from_the_model(client, mocker):
    client._sessions[_BUILTIN] = fake_session(
        mocker, [tool("read_file"), tool("_preview_edit"), tool("_file_exists")])
    names = [s["function"]["name"] for s in await client.list_llm_tools()]
    assert names == ["read_file"]


async def test_custom_server_tools_are_namespaced(client, mocker):
    client._sessions["docs"] = fake_session(mocker, [tool("search")])
    names = [s["function"]["name"] for s in await client.list_llm_tools()]
    assert names == ["docs__search"]


async def test_custom_server_underscore_tools_are_not_filtered(client, mocker):
    """Only the built-in server's underscore tools are internal."""
    client._sessions["docs"] = fake_session(mocker, [tool("_odd")])
    assert [s["function"]["name"] for s in await client.list_llm_tools()] == ["docs___odd"]


async def test_tool_owner_routing_map_is_rebuilt_each_call(client, mocker):
    client._sessions["docs"] = fake_session(mocker, [tool("search")])
    await client.list_llm_tools()
    assert client._tool_owner["docs__search"] == ("docs", "search")

    client._sessions["docs"] = fake_session(mocker, [tool("other")])
    await client.list_llm_tools()
    assert "docs__search" not in client._tool_owner  # stale entry dropped


# ---------------- deferred loading + search_tools ----------------

async def test_deferred_server_tools_are_withheld_and_search_tools_offered(client, mocker):
    client._sessions["docs"] = fake_session(mocker, [tool("a"), tool("b")])
    client._deferred_servers.add("docs")
    names = [s["function"]["name"] for s in await client.list_llm_tools()]
    assert names == ["search_tools"]
    assert set(client._deferred_tools) == {"docs__a", "docs__b"}


async def test_no_search_tools_when_nothing_is_deferred(client, mocker):
    client._sessions["docs"] = fake_session(mocker, [tool("a")])
    assert "search_tools" not in [s["function"]["name"] for s in await client.list_llm_tools()]


async def test_revealed_tools_reappear_in_the_next_listing(client, mocker):
    client._sessions["docs"] = fake_session(mocker, [tool("a"), tool("b")])
    client._deferred_servers.add("docs")
    await client.list_llm_tools()
    await client.search_mcp_tools("")          # empty query reveals everything
    names = [s["function"]["name"] for s in await client.list_llm_tools()]
    assert "docs__a" in names and "docs__b" in names


async def test_search_tools_empty_query_reveals_all(client, mocker):
    client._sessions["docs"] = fake_session(mocker, [tool("alpha"), tool("beta")])
    client._deferred_servers.add("docs")
    await client.list_llm_tools()
    out = await client.search_mcp_tools("")
    assert "Loaded 2 tool(s)" in out
    assert client._revealed == {"docs__alpha", "docs__beta"}


async def test_search_tools_keyword_match_when_embeddings_disabled(client, mocker):
    client.embedding_model = ""
    client._sessions["docs"] = fake_session(
        mocker, [tool("weather", "Get the forecast"), tool("stocks", "Share prices")])
    client._deferred_servers.add("docs")
    await client.list_llm_tools()
    out = await client.search_mcp_tools("forecast")
    assert "docs__weather" in out and "docs__stocks" not in out


async def test_search_tools_keyword_requires_every_term(client, mocker):
    client.embedding_model = ""
    client._sessions["d"] = fake_session(mocker, [tool("a", "alpha beta"), tool("b", "alpha only")])
    client._deferred_servers.add("d")
    await client.list_llm_tools()
    out = await client.search_mcp_tools("alpha beta")
    assert "d__a" in out and "d__b" not in out


async def test_search_tools_reports_no_match_and_lists_remaining(client, mocker):
    client.embedding_model = ""
    client._sessions["d"] = fake_session(mocker, [tool("a", "alpha")])
    client._deferred_servers.add("d")
    await client.list_llm_tools()
    out = await client.search_mcp_tools("nothing-like-this")
    assert "No deferred tools matched" in out and "d__a" in out
    assert client._revealed == set()


async def test_search_tools_with_nothing_deferred(client):
    assert "already loaded" in await client.search_mcp_tools("x")


async def test_search_tools_semantic_path_ranks_by_similarity(client, mocker):
    client.embedding_model = "some-remote-model"
    client._sessions["d"] = fake_session(mocker, [tool("near", "close"), tool("far", "distant")])
    client._deferred_servers.add("d")
    await client.list_llm_tools()

    # query vector matches "near" exactly and is orthogonal to "far"
    vectors = {"query": [1.0, 0.0], "d__near: close": [1.0, 0.0], "d__far: distant": [0.0, 1.0]}
    mocker.patch.object(client, "_embed_texts",
                        mocker.AsyncMock(side_effect=lambda texts, task_type: [vectors[t] for t in texts]))
    out = await client.search_mcp_tools("query")
    assert "d__near" in out and "d__far" not in out


async def test_search_tools_falls_back_to_keywords_when_embedding_unavailable(client, mocker):
    client.embedding_model = "broken-model"
    client._sessions["d"] = fake_session(mocker, [tool("weather", "forecast")])
    client._deferred_servers.add("d")
    await client.list_llm_tools()
    mocker.patch.object(client, "_embed_texts",
                        mocker.AsyncMock(side_effect=EmbeddingUnavailable("no backend")))
    out = await client.search_mcp_tools("forecast")
    assert "fell back to keyword match" in out and "d__weather" in out


# ---------------- call_tool routing ----------------

async def test_call_tool_routes_to_the_owning_server(client, mocker):
    builtin = fake_session(mocker, [tool("read_file")])
    docs = fake_session(mocker, [tool("search")])
    for s in (builtin, docs):
        s.call_tool.return_value = mocker.Mock(isError=False, content=[mocker.Mock(text="ok")])
    client._sessions.update({_BUILTIN: builtin, "docs": docs})
    await client.list_llm_tools()

    await client.call_tool("docs__search", {"q": 1})
    docs.call_tool.assert_awaited_once_with("search", {"q": 1})  # un-namespaced on the wire
    builtin.call_tool.assert_not_awaited()


async def test_call_tool_defaults_unknown_names_to_builtin(client, mocker):
    """Internal _preview_* tools aren't in _tool_owner (list_llm_tools filters
    them), so they must fall through to the built-in session."""
    builtin = fake_session(mocker)
    builtin.call_tool.return_value = mocker.Mock(isError=False, content=[mocker.Mock(text="OK\ndiff")])
    client._sessions[_BUILTIN] = builtin
    await client.call_tool("_preview_edit", {"path": "x"})
    builtin.call_tool.assert_awaited_once_with("_preview_edit", {"path": "x"})


async def test_call_tool_marks_server_errors(client, mocker):
    s = fake_session(mocker, [tool("boom")])
    s.call_tool.return_value = mocker.Mock(isError=True, content=[mocker.Mock(text="it failed")])
    client._sessions["d"] = s
    await client.list_llm_tools()
    assert await client.call_tool("d__boom", {}) == "ERROR: it failed"


async def test_call_tool_concatenates_text_blocks_and_ignores_others(client, mocker):
    s = fake_session(mocker, [tool("multi")])
    non_text = mocker.Mock(spec=[])  # no .text attribute
    s.call_tool.return_value = mocker.Mock(
        isError=False, content=[mocker.Mock(text="a"), non_text, mocker.Mock(text="b")])
    client._sessions["d"] = s
    await client.list_llm_tools()
    assert await client.call_tool("d__multi", {}) == "ab"


async def test_call_tool_refuses_still_deferred_tool(client, mocker):
    client._sessions["d"] = fake_session(mocker, [tool("hidden")])
    client._deferred_servers.add("d")
    await client.list_llm_tools()
    out = await client.call_tool("d__hidden", {})
    assert out.startswith("ERROR:") and "search_tools" in out


async def test_call_tool_dispatches_search_tools_internally(client, mocker):
    client._sessions["d"] = fake_session(mocker, [tool("a")])
    client._deferred_servers.add("d")
    await client.list_llm_tools()
    assert "Loaded 1 tool(s)" in await client.call_tool("search_tools", {"query": ""})


# ---------------- prompts ----------------

def prompt(name, description="", args=()):
    p = type("P", (), {})()
    p.name = name
    p.description = description
    p.arguments = [type("A", (), {"name": a[0], "description": "", "required": a[1]})() for a in args]
    return p


async def test_list_prompts_namespaced_with_colon(client, mocker):
    client._sessions["docs"] = fake_session(
        mocker, prompts=[prompt("summarize", "Sum it up", [("path", True), ("style", False)])])
    out = await client.list_prompts()
    assert set(out) == {"docs:summarize"}
    assert out["docs:summarize"]["description"] == "Sum it up"
    assert out["docs:summarize"]["arguments"] == [
        {"name": "path", "description": "", "required": True},
        {"name": "style", "description": "", "required": False},
    ]


async def test_list_prompts_skips_servers_without_the_capability(client, mocker):
    good = fake_session(mocker, prompts=[prompt("p")])
    bad = fake_session(mocker)
    bad.list_prompts.side_effect = Exception("method not found")
    client._sessions.update({"good": good, "bad": bad})
    assert set(await client.list_prompts()) == {"good:p"}


async def test_get_prompt_flattens_messages(client, mocker):
    s = fake_session(mocker, prompts=[prompt("p")])
    s.get_prompt.return_value = mocker.Mock(messages=[
        mocker.Mock(content=mocker.Mock(text="first")),
        mocker.Mock(content=mocker.Mock(text="second")),
    ])
    client._sessions["d"] = s
    await client.list_prompts()
    assert await client.get_prompt("d:p", {"a": "1"}) == "first\n\nsecond"
    s.get_prompt.assert_awaited_once_with("p", {"a": "1"})


async def test_get_prompt_passes_none_for_empty_arguments(client, mocker):
    s = fake_session(mocker, prompts=[prompt("p")])
    s.get_prompt.return_value = mocker.Mock(messages=[])
    client._sessions["d"] = s
    await client.list_prompts()
    await client.get_prompt("d:p", {})
    assert s.get_prompt.await_args.args[1] is None


async def test_get_prompt_unknown_name_raises(client):
    with pytest.raises(ValueError, match="Unknown prompt"):
        await client.get_prompt("nope:missing")


# ---------------- resources ----------------

def resource(uri, name="n", description="", mime=None, size=None):
    r = type("R", (), {})()
    r.uri = uri
    r.name = name
    r.description = description
    r.mimeType = mime
    r.size = size
    return r


def template(uri_template, name="t", description=""):
    r = type("RT", (), {})()
    r.uriTemplate = uri_template
    r.name = name
    r.description = description
    r.mimeType = None
    r.size = None
    return r


async def test_list_resources_keyed_by_uri_with_metadata(client, mocker):
    client._sessions["docs"] = fake_session(
        mocker, resources=[resource("file:///a.md", "std", "Standards", "text/markdown", 42)])
    out = await client.list_resources()
    entry = out["file:///a.md"]
    assert entry["server"] == "docs" and entry["name"] == "std"
    assert entry["description"] == "Standards" and entry["mime_type"] == "text/markdown"
    assert entry["size"] == 42 and entry["template"] is False


async def test_list_resources_excludes_templates_by_default(client, mocker):
    client._sessions["d"] = fake_session(
        mocker, resources=[resource("file:///a.md")], templates=[template("file:///{x}.log")])
    assert set(await client.list_resources()) == {"file:///a.md"}


async def test_list_resources_includes_templates_on_request(client, mocker):
    client._sessions["d"] = fake_session(
        mocker, resources=[resource("file:///a.md")], templates=[template("file:///{x}.log")])
    out = await client.list_resources(include_templates=True)
    assert out["file:///{x}.log"]["template"] is True
    assert "file:///{x}.log" not in client._resource_owner  # not readable directly


async def test_list_resources_skips_servers_without_the_capability(client, mocker):
    bad = fake_session(mocker)
    bad.list_resources.side_effect = Exception("unsupported")
    client._sessions.update({"bad": bad, "good": fake_session(mocker, resources=[resource("x://1")])})
    assert set(await client.list_resources()) == {"x://1"}


async def test_duplicate_uri_first_server_wins_and_records_shadowing(client, mocker):
    client._sessions["first"] = fake_session(mocker, resources=[resource("file:///dup.md")])
    client._sessions["second"] = fake_session(mocker, resources=[resource("file:///dup.md")])
    out = await client.list_resources()
    assert out["file:///dup.md"]["server"] == "first"
    assert out["file:///dup.md"]["shadowed_by"] == ["second"]
    assert client._resource_owner["file:///dup.md"] == "first"


async def test_read_resource_routes_by_uri(client, mocker):
    s = fake_session(mocker, resources=[resource("file:///a.md")])
    s.read_resource.return_value = mocker.Mock(contents=[mocker.Mock(text="body", spec=["text"])])
    client._sessions["docs"] = s
    await client.list_resources()
    assert await client.read_resource("file:///a.md") == "body"
    s.read_resource.assert_awaited_once_with("file:///a.md")


async def test_read_resource_joins_multiple_text_blocks(client, mocker):
    s = fake_session(mocker, resources=[resource("x://1")])
    s.read_resource.return_value = mocker.Mock(contents=[
        mocker.Mock(text="one", spec=["text"]), mocker.Mock(text="two", spec=["text"])])
    client._sessions["d"] = s
    await client.list_resources()
    assert await client.read_resource("x://1") == "one\n\ntwo"


async def test_read_resource_replaces_binary_with_a_size_marker(client, mocker):
    import base64
    blob = mocker.Mock(spec=["blob", "mimeType"])
    blob.blob = base64.b64encode(b"x" * 108).decode()
    blob.mimeType = "image/png"
    s = fake_session(mocker, resources=[resource("file:///logo.png")])
    s.read_resource.return_value = mocker.Mock(contents=[blob])
    client._sessions["d"] = s
    await client.list_resources()
    out = await client.read_resource("file:///logo.png")
    assert out == "[binary image/png, 108 bytes — not shown]"


async def test_read_resource_handles_undecodable_blob(client, mocker):
    blob = mocker.Mock(spec=["blob", "mimeType"])
    blob.blob = "!!!not-base64!!!"
    blob.mimeType = None
    s = fake_session(mocker, resources=[resource("x://1")])
    s.read_resource.return_value = mocker.Mock(contents=[blob])
    client._sessions["d"] = s
    await client.list_resources()
    assert "application/octet-stream" in await client.read_resource("x://1")


async def test_read_resource_explicit_server_overrides_routing(client, mocker):
    first = fake_session(mocker, resources=[resource("file:///dup.md")])
    second = fake_session(mocker, resources=[resource("file:///dup.md")])
    first.read_resource.return_value = mocker.Mock(contents=[mocker.Mock(text="from-first", spec=["text"])])
    second.read_resource.return_value = mocker.Mock(contents=[mocker.Mock(text="from-second", spec=["text"])])
    client._sessions.update({"first": first, "second": second})
    await client.list_resources()
    assert await client.read_resource("file:///dup.md") == "from-first"
    assert await client.read_resource("file:///dup.md", server="second") == "from-second"


async def test_read_resource_unknown_uri_and_server_raise(client, mocker):
    client._sessions["d"] = fake_session(mocker)
    with pytest.raises(ValueError, match="Unknown resource"):
        await client.read_resource("file:///nope.md")
    with pytest.raises(ValueError, match="Unknown MCP server"):
        await client.read_resource("file:///nope.md", server="ghost")


# ---------------- resource tools exposed to the model ----------------

async def test_resource_tools_offered_only_when_resources_exist(client, mocker):
    client._sessions[_BUILTIN] = fake_session(mocker, [tool("read_file")])
    assert "read_resource" not in [s["function"]["name"] for s in await client.list_llm_tools()]

    client._resources = {"file:///a.md": {"template": False, "name": "", "description": "",
                                          "mime_type": "", "size": None, "shadowed_by": []}}
    names = [s["function"]["name"] for s in await client.list_llm_tools()]
    assert "list_resources" in names and "read_resource" in names


async def test_call_tool_list_resources_renders_catalog(client, mocker):
    client._sessions["d"] = fake_session(
        mocker, resources=[resource("file:///a.md", "std", "Standards", "text/markdown")])
    out = await client.call_tool("list_resources", {})
    assert "file:///a.md" in out and "text/markdown" in out and "Standards" in out


async def test_call_tool_list_resources_when_none(client, mocker):
    client._sessions["d"] = fake_session(mocker)
    assert "no resources" in await client.call_tool("list_resources", {})


async def test_call_tool_read_resource_success_and_errors(client, mocker):
    s = fake_session(mocker, resources=[resource("file:///a.md")])
    s.read_resource.return_value = mocker.Mock(contents=[mocker.Mock(text="content", spec=["text"])])
    client._sessions["d"] = s
    await client.list_resources()

    assert await client.call_tool("read_resource", {"uri": "file:///a.md"}) == "content"
    assert (await client.call_tool("read_resource", {})).startswith("ERROR:")
    assert (await client.call_tool("read_resource", {"uri": "x://none"})).startswith("ERROR:")


async def test_call_tool_read_resource_wraps_unexpected_failures(client, mocker):
    s = fake_session(mocker, resources=[resource("x://1")])
    s.read_resource.side_effect = RuntimeError("transport died")
    client._sessions["d"] = s
    await client.list_resources()
    out = await client.call_tool("read_resource", {"uri": "x://1"})
    assert out.startswith("ERROR:") and "transport died" in out


# ---------------- status / names / spec resolution ----------------

async def test_server_status_reports_connected_and_failed(client, mocker):
    client._server_specs[_BUILTIN] = {"command": "python", "args": ["-m", "omni.mcp_server"]}
    client._server_specs["docs"] = {"url": "https://h/mcp"}
    client._server_specs["dead"] = {"command": "nope"}
    client._sessions[_BUILTIN] = fake_session(mocker, [tool("read_file")])
    client._sessions["docs"] = fake_session(mocker, [tool("a"), tool("b")])
    client._connected_at[_BUILTIN] = client._connected_at["docs"] = 0.0
    client._connect_errors["dead"] = "boom"
    client._deferred_servers.add("docs")
    await client.list_llm_tools()

    by_name = {e["name"]: e for e in client.server_status()}
    assert by_name["built-in"]["connected"] is True
    assert by_name["built-in"]["tool_count"] == 1
    assert by_name["docs"]["deferred"] is True and by_name["docs"]["tool_count"] == 2
    assert by_name["docs"]["target"] == "https://h/mcp"
    assert by_name["dead"]["connected"] is False and by_name["dead"]["error"] == "boom"
    assert by_name["dead"]["connected_for"] is None


def test_server_names_puts_builtin_first(client):
    client._server_specs.update({_BUILTIN: {}, "z": {}, "a": {}})
    assert client.server_names() == ["built-in", "z", "a"]


def test_builtin_spec_points_at_the_package_module(client):
    spec = client._builtin_spec()
    assert spec["args"][-1].endswith("mcp_server")
    assert spec["env"]["AGENT_PROJECT_ROOT"] == client.project_root


def test_resolve_spec_prefers_cli_over_config_file(tmp_path):
    import json
    cfg_file = tmp_path / "s.json"
    cfg_file.write_text(json.dumps({"mcpServers": {"d": {"command": "from-file"}}}))
    c = MCPToolClient(str(tmp_path), mcp_config_path=str(cfg_file),
                      extra_servers={"d": {"command": "from-cli"}},
                      mcp_log_path=str(tmp_path / "l.log"))
    assert c._resolve_spec("d")["command"] == "from-cli"


def test_resolve_spec_rereads_the_config_file(tmp_path):
    """A restart must pick up edits to the settings file, not just code."""
    import json
    cfg_file = tmp_path / "s.json"
    cfg_file.write_text(json.dumps({"mcpServers": {"d": {"command": "old"}}}))
    c = MCPToolClient(str(tmp_path), mcp_config_path=str(cfg_file),
                      mcp_log_path=str(tmp_path / "l.log"))
    assert c._resolve_spec("d")["command"] == "old"
    cfg_file.write_text(json.dumps({"mcpServers": {"d": {"command": "new"}}}))
    assert c._resolve_spec("d")["command"] == "new"


def test_resolve_spec_falls_back_to_startup_spec(tmp_path):
    c = MCPToolClient(str(tmp_path), mcp_config_path=str(tmp_path / "gone.json"),
                      mcp_log_path=str(tmp_path / "l.log"))
    c._server_specs["d"] = {"command": "remembered"}
    assert c._resolve_spec("d")["command"] == "remembered"


def test_resolve_spec_for_builtin_rebuilds_it(client):
    assert client._resolve_spec(_BUILTIN)["args"][-1].endswith("mcp_server")


# ---------------- restart bookkeeping ----------------

async def test_restart_unknown_server_raises_listing_known(client):
    client._server_specs[_BUILTIN] = {}
    client._server_specs["docs"] = {}
    with pytest.raises(ValueError, match="built-in, docs"):
        await client.restart_server("ghost")


@pytest.mark.parametrize("alias", ["built-in", "builtin", _BUILTIN])
async def test_restart_accepts_builtin_aliases(client, mocker, alias):
    client._server_specs[_BUILTIN] = client._builtin_spec()
    mocker.patch.object(client, "_stop_server", mocker.AsyncMock())
    mocker.patch.object(client, "_connect", mocker.AsyncMock(return_value=fake_session(mocker)))
    mocker.patch.object(client, "list_resources", mocker.AsyncMock(return_value={}))
    entry = await client.restart_server(alias)
    assert entry["name"] == "built-in" and entry["connected"] is True


async def test_restart_stops_before_reconnecting(client, mocker):
    """The old process must be gone before the replacement starts."""
    order = []
    client._server_specs["d"] = {"command": "x"}
    mocker.patch.object(client, "_stop_server",
                        mocker.AsyncMock(side_effect=lambda n: order.append("stop")))
    mocker.patch.object(client, "_connect", mocker.AsyncMock(
        side_effect=lambda n, s: order.append("connect") or fake_session(mocker)))
    mocker.patch.object(client, "list_resources", mocker.AsyncMock(return_value={}))
    await client.restart_server("d")
    assert order == ["stop", "connect"]


async def test_restart_clears_stale_state_and_records_success(client, mocker):
    client._server_specs["d"] = {"command": "x"}
    client._connect_errors["d"] = "old failure"
    client._tool_embeddings.update({"d__a": [1.0], "other__b": [2.0]})
    mocker.patch.object(client, "_stop_server", mocker.AsyncMock())
    mocker.patch.object(client, "_connect", mocker.AsyncMock(return_value=fake_session(mocker, [tool("a")])))
    mocker.patch.object(client, "list_resources", mocker.AsyncMock(return_value={}))

    entry = await client.restart_server("d")
    assert entry["connected"] is True and entry["error"] is None
    assert "d" not in client._connect_errors
    assert "d__a" not in client._tool_embeddings      # stale embedding dropped
    assert "other__b" in client._tool_embeddings      # other servers untouched
    assert client._tool_owner["d__a"] == ("d", "a")   # routing refreshed


async def test_restart_records_failure_without_raising(client, mocker):
    client._server_specs["d"] = {"command": "x"}
    mocker.patch.object(client, "_stop_server", mocker.AsyncMock())
    mocker.patch.object(client, "_connect", mocker.AsyncMock(side_effect=RuntimeError("still broken")))
    mocker.patch.object(client, "list_resources", mocker.AsyncMock(return_value={}))

    entry = await client.restart_server("d")
    assert entry["connected"] is False and "still broken" in entry["error"]
    assert "d" not in client._sessions


async def test_restart_updates_deferred_flag_from_the_new_spec(client, mocker):
    client._server_specs["d"] = {"command": "x", "defer": True}
    mocker.patch.object(client, "_stop_server", mocker.AsyncMock())
    mocker.patch.object(client, "_connect", mocker.AsyncMock(return_value=fake_session(mocker)))
    mocker.patch.object(client, "list_resources", mocker.AsyncMock(return_value={}))
    mocker.patch.object(client, "_resolve_spec", lambda n: {"command": "x", "defer": True})
    await client.restart_server("d")
    assert "d" in client._deferred_servers

    mocker.patch.object(client, "_resolve_spec", lambda n: {"command": "x"})  # defer removed
    await client.restart_server("d")
    assert "d" not in client._deferred_servers


async def test_stop_server_is_a_noop_for_unknown_name(client):
    await client._stop_server("never-started")   # must not raise


async def test_stop_server_signals_and_awaits_its_task(client):
    started = asyncio.Event()

    async def fake_serve(shutdown):
        started.set()
        await shutdown.wait()

    shutdown = asyncio.Event()
    task = asyncio.ensure_future(fake_serve(shutdown))
    await started.wait()
    client._server_tasks["d"] = (task, shutdown)

    await client._stop_server("d")
    assert task.done() and "d" not in client._server_tasks


async def test_stop_server_swallows_benign_teardown_race(client):
    from anyio import BrokenResourceError

    async def racy(shutdown):
        await shutdown.wait()
        raise BrokenResourceError()

    shutdown = asyncio.Event()
    client._server_tasks["d"] = (asyncio.ensure_future(racy(shutdown)), shutdown)
    await client._stop_server("d")   # must not propagate


async def test_stop_server_propagates_real_errors(client):
    async def broken(shutdown):
        await shutdown.wait()
        raise RuntimeError("genuine failure")

    shutdown = asyncio.Event()
    client._server_tasks["d"] = (asyncio.ensure_future(broken(shutdown)), shutdown)
    with pytest.raises(RuntimeError, match="genuine failure"):
        await client._stop_server("d")


# ---------------- preview / file_exists convenience wrappers ----------------

async def test_preview_edit_parses_ok_and_error(client, mocker):
    builtin = fake_session(mocker)
    client._sessions[_BUILTIN] = builtin

    builtin.call_tool.return_value = mocker.Mock(isError=False, content=[mocker.Mock(text="OK\nthe diff")])
    assert await client.preview_edit("f", "a", "b") == (True, "the diff")

    builtin.call_tool.return_value = mocker.Mock(isError=False, content=[mocker.Mock(text="ERROR\nnope")])
    assert await client.preview_edit("f", "a", "b") == (False, "nope")


async def test_preview_write_parses_new_and_diff(client, mocker):
    builtin = fake_session(mocker)
    client._sessions[_BUILTIN] = builtin

    builtin.call_tool.return_value = mocker.Mock(isError=False, content=[mocker.Mock(text="NEW\n+a")])
    assert await client.preview_write("f", "a") == (True, "+a")

    builtin.call_tool.return_value = mocker.Mock(isError=False, content=[mocker.Mock(text="DIFF\n-a\n+b")])
    assert await client.preview_write("f", "b", overwrite=True) == (False, "-a\n+b")


async def test_file_exists_parses_boolean(client, mocker):
    builtin = fake_session(mocker)
    client._sessions[_BUILTIN] = builtin
    builtin.call_tool.return_value = mocker.Mock(isError=False, content=[mocker.Mock(text="true")])
    assert await client.file_exists("x") is True
    builtin.call_tool.return_value = mocker.Mock(isError=False, content=[mocker.Mock(text="false")])
    assert await client.file_exists("x") is False
