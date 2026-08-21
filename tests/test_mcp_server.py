"""mcp_server.py — the MCP wrapper around Tools.

Verifies the exposed tool surface (names, which are internal, docstrings the
model reads) and that each wrapper forwards to the matching Tools method,
with `impl` mocked so no real file/git/shell work happens.
"""

import pytest

from omni import mcp_server as srv


# ---------------- exposed surface ----------------

async def tool_names():
    return {t.name for t in await srv.mcp.list_tools()}


async def test_public_tools_are_exposed():
    names = await tool_names()
    expected = {
        "read_file", "list_dir", "search_files", "glob_files",
        "git_diff", "git_status", "git_log", "git_show", "git_branch", "git_fetch",
        "git_add", "git_commit", "git_pull", "git_push",
        "write_file", "edit_file", "run_shell", "save_memory",
    }
    assert expected <= names


async def test_internal_tools_are_underscore_prefixed():
    """mcp_client filters built-in tools starting with "_" out of what the
    model sees, so the preview/existence helpers must keep that prefix."""
    names = await tool_names()
    assert {"_preview_edit", "_preview_write", "_file_exists"} <= names


async def test_every_tool_has_a_description():
    for tool in await srv.mcp.list_tools():
        assert (tool.description or "").strip(), f"{tool.name} has no docstring"


async def test_write_tools_document_their_guardrails():
    by_name = {t.name: t.description for t in await srv.mcp.list_tools()}
    assert "overwrite" in by_name["write_file"].lower()
    assert "unique" in by_name["edit_file"].lower()


async def test_server_is_named():
    assert srv.mcp.name == "omni-tools"


# ---------------- delegation ----------------

@pytest.fixture
def impl(mocker):
    return mocker.patch.object(srv, "impl")


@pytest.mark.parametrize("fn,args,method", [
    ("read_file", ("a.py", 1, 5), "read_file"),
    ("list_dir", ("sub",), "list_dir"),
    ("search_files", ("pat", ".", "*.py", True), "search_files"),
    ("glob_files", ("**/*.py", "."), "glob_files"),
    ("git_diff", (".",), "git_diff"),
    ("git_status", (".",), "git_status"),
    ("git_log", (".", 5), "git_log"),
    ("git_show", ("HEAD", "."), "git_show"),
    ("git_branch", (".",), "git_branch"),
    ("git_fetch", ("origin",), "git_fetch"),
    ("git_add", ("a.py",), "git_add"),
    ("git_commit", ("msg",), "git_commit"),
    ("git_pull", ("origin", "main"), "git_pull"),
    ("git_push", ("origin", "main"), "git_push"),
    ("write_file", ("a.py", "body", True), "write_file"),
    ("edit_file", ("a.py", "old", "new"), "edit_file"),
    ("run_shell", ("ls",), "run_shell"),
    ("save_memory", ("a note",), "save_memory"),
])
def test_wrapper_forwards_to_tools(impl, fn, args, method):
    getattr(srv, fn)(*args)
    getattr(impl, method).assert_called_once_with(*args)


def test_preview_edit_encodes_ok_and_error(impl):
    impl.preview_edit.return_value = (True, "the diff")
    assert srv._preview_edit("a.py", "x", "y") == "OK\nthe diff"

    impl.preview_edit.return_value = (False, "not unique")
    assert srv._preview_edit("a.py", "x", "y") == "ERROR\nnot unique"


def test_preview_write_encodes_new_and_diff(impl):
    impl.preview_write.return_value = (True, "+added")
    assert srv._preview_write("a.py", "body") == "NEW\n+added"

    impl.preview_write.return_value = (False, "-old\n+new")
    assert srv._preview_write("a.py", "body", True) == "DIFF\n-old\n+new"


def test_file_exists_encodes_boolean(impl):
    impl.file_exists.return_value = True
    assert srv._file_exists("a.py") == "true"
    impl.file_exists.return_value = False
    assert srv._file_exists("a.py") == "false"


# ---------------- project-root scoping ----------------

def test_project_root_comes_from_the_environment(monkeypatch, tmp_path):
    """The client passes AGENT_PROJECT_ROOT when spawning this server, so the
    module must read it at import time."""
    import importlib
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    reloaded = importlib.reload(srv)
    try:
        assert reloaded.cfg.project_root == str(tmp_path)
        assert reloaded.impl.cfg.project_root == str(tmp_path)
    finally:
        monkeypatch.delenv("AGENT_PROJECT_ROOT", raising=False)
        importlib.reload(srv)


def test_scoping_is_enforced_through_the_wrapper(monkeypatch, tmp_path):
    """An escape attempt must raise from the real Tools, not be silently
    served — checked end-to-end through the MCP-facing function."""
    import importlib
    from omni.tools import PathScopeError
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    reloaded = importlib.reload(srv)
    try:
        with pytest.raises(PathScopeError):
            reloaded.read_file("../../etc/passwd")
    finally:
        monkeypatch.delenv("AGENT_PROJECT_ROOT", raising=False)
        importlib.reload(srv)
