"""Shared fixtures.

Everything here is hermetic: each test gets its own project root, SQLite
file, and log paths under tmp_path, so nothing touches the developer's real
~/.omni-coder settings, agent_sessions.db, or working tree.
"""

import sys
import textwrap

import pytest

from omni.config import AgentConfig


@pytest.fixture
def project_root(tmp_path):
    """An isolated directory to act as --project-root."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "hello.py").write_text("def hello():\n    return 1\n")
    (root / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    sub = root / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text("import os\n\n\ndef helper():\n    return os.sep\n")
    return root


@pytest.fixture
def cfg(tmp_path, project_root):
    """AgentConfig with every on-disk path redirected into tmp_path."""
    return AgentConfig(
        project_root=str(project_root),
        log_path=str(tmp_path / "agent_run.log"),
        db_path=str(tmp_path / "sessions.db"),
        mcp_log_path=str(tmp_path / "mcp_servers.log"),
        parse_intent=False,
        embedding_model="",
    )


@pytest.fixture
def tools(cfg):
    from omni.tools import Tools
    return Tools(cfg)


def write_mcp_server(path, name="toy", tools_src="", prompts_src="", resources_src=""):
    """Write a runnable FastMCP stdio server script, for the live MCP tests."""
    path.write_text(textwrap.dedent(f'''
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP({name!r})
        {textwrap.indent(textwrap.dedent(tools_src), "        ").strip()}
        {textwrap.indent(textwrap.dedent(prompts_src), "        ").strip()}
        {textwrap.indent(textwrap.dedent(resources_src), "        ").strip()}
        if __name__ == "__main__":
            mcp.run()
    '''))
    return path


@pytest.fixture
def mcp_server_factory(tmp_path):
    """Returns a callable that writes a toy MCP server and hands back the
    {"command", "args"} spec for it."""
    counter = {"n": 0}

    def make(tools_src="", prompts_src="", resources_src="", name="toy"):
        counter["n"] += 1
        path = tmp_path / f"srv_{counter['n']}.py"
        write_mcp_server(path, name, tools_src, prompts_src, resources_src)
        return {"command": sys.executable, "args": [str(path)]}, path

    return make
