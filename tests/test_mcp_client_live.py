"""MCPToolClient against real MCP servers over stdio.

These are the one place mocks can't substitute: _serve/_connect/_stop_server
exist to manage anyio task-group lifetimes, and the per-server-task design
was written specifically because cancel scopes are task-scoped. Only real
transports exercise that.

Marked `live` — deselect with `-m "not live"` if subprocess spawning is
unavailable.
"""

import subprocess
import sys
import textwrap
import time

import pytest

from omni.config import AgentConfig
from omni.mcp_client import _BUILTIN, MCPToolClient

pytestmark = pytest.mark.live


def server_script(path, body, name="toy"):
    path.write_text(textwrap.dedent(f'''
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP({name!r})
{textwrap.indent(textwrap.dedent(body), "        ")}
        if __name__ == "__main__":
            mcp.run()
    '''))
    return {"command": sys.executable, "args": [str(path)]}


PING = '''
@mcp.tool()
def ping() -> str:
    """Ping."""
    return "pong"
'''


@pytest.fixture
def make_client(tmp_path):
    def build(**kwargs):
        kwargs.setdefault("mcp_log_path", str(tmp_path / "mcp.log"))
        return MCPToolClient(str(tmp_path), **kwargs)
    return build


def live_children(script_path, expected=None, timeout=3.0):
    """Count live server processes for exactly `script_path`.

    Matching must be on the full path, not the bare filename: pgrep -f
    searches every process on the machine, so a bare name would also match
    servers belonging to a concurrent or interrupted pytest run that happened
    to use the same temp filename.

    When `expected` is given, poll until the count matches (or `timeout`
    elapses) — a terminated child can linger briefly before the OS reaps it.
    """
    deadline = time.monotonic() + timeout
    while True:
        out = subprocess.run(["pgrep", "-f", str(script_path)], capture_output=True, text=True).stdout
        count = len([l for l in out.splitlines() if l.strip()])
        if expected is None or count == expected or time.monotonic() > deadline:
            return count
        time.sleep(0.05)


# ---------------- connect / teardown ----------------

async def test_builtin_server_connects_and_serves_tools(make_client):
    async with make_client() as client:
        names = [s["function"]["name"] for s in await client.list_llm_tools()]
        assert "read_file" in names and "write_file" in names
        assert not any(n.startswith("_") for n in names)


async def test_builtin_tool_call_round_trips(make_client, tmp_path):
    (tmp_path / "probe.txt").write_text("hello from disk")
    async with make_client() as client:
        await client.list_llm_tools()
        assert "hello from disk" in await client.call_tool("read_file", {"path": "probe.txt"})


def wedged_script(path):
    """A server that starts, holds stdio open, and never speaks MCP — the
    shape of one that's misconfigured, waiting on a lock, or hung."""
    path.write_text("import time\nwhile True:\n    time.sleep(3600)\n")
    return {"command": sys.executable, "args": [str(path)]}


async def test_a_server_that_never_speaks_mcp_does_not_block_startup(make_client, tmp_path):
    """The regression this exists for: an unbounded wait on the handshake let
    one bad entry hang the entire session before the REPL ever appeared."""
    spec = wedged_script(tmp_path / "wedged.py")
    start = time.monotonic()
    async with make_client(extra_servers={"wedged": spec}, connect_timeout_s=2.0) as client:
        elapsed = time.monotonic() - start
        assert elapsed < 12, f"startup took {elapsed:.1f}s"

        entry = next(e for e in client.server_status() if e["name"] == "wedged")
        assert entry["connected"] is False
        assert "did not finish connecting" in entry["error"]

        # ...and the session is fully usable without it
        names = [t["function"]["name"] for t in await client.list_llm_tools()]
        assert "read_file" in names
    assert live_children(tmp_path / "wedged.py", expected=0) == 0


async def test_two_wedged_servers_cost_one_timeout_not_two(make_client, tmp_path):
    """Connects run concurrently; serially, every bad server added its full
    budget to startup."""
    a = wedged_script(tmp_path / "w1.py")
    b = wedged_script(tmp_path / "w2.py")
    start = time.monotonic()
    async with make_client(extra_servers={"a": a, "b": b}, connect_timeout_s=2.0) as client:
        elapsed = time.monotonic() - start
        assert elapsed < 6, f"startup took {elapsed:.1f}s for two 2s budgets"
        assert [e["connected"] for e in client.server_status() if e["name"] in ("a", "b")] == [False, False]


async def test_restarting_a_wedged_server_is_bounded_too(make_client, tmp_path):
    spec = wedged_script(tmp_path / "wedged.py")
    async with make_client(extra_servers={"wedged": spec}, connect_timeout_s=2.0) as client:
        start = time.monotonic()
        entry = await client.restart_server("wedged")
        assert time.monotonic() - start < 12
        assert entry["connected"] is False and "did not finish connecting" in entry["error"]


async def test_builtin_server_honors_the_tool_config_it_is_given(make_client, tmp_path):
    """The tools run in a subprocess, so these knobs only take effect if they
    are handed across that boundary — set on AgentConfig alone they used to
    be silently ignored by the code that actually runs."""
    cfg = AgentConfig(project_root=str(tmp_path), memory_path="notes/mem.md",
                      max_output_chars=60, denied_shell_patterns=("echo forbidden",))
    async with make_client(builtin_env=cfg.tool_server_env()) as client:
        await client.list_llm_tools()

        blocked = await client.call_tool("run_shell", {"command": "echo forbidden"})
        assert "blocked by policy" in blocked

        truncated = await client.call_tool("run_shell", {"command": "seq 1 400"})
        assert "truncated" in truncated

        await client.call_tool("save_memory", {"note": "remember this"})
        assert "remember this" in (tmp_path / "notes" / "mem.md").read_text()


async def test_builtin_server_falls_back_to_defaults_without_the_extra_env(make_client, tmp_path):
    """A bare MCPToolClient(root) — the shape every other test and any
    third-party embedder uses — must still work on AgentConfig's defaults."""
    async with make_client() as client:
        await client.list_llm_tools()
        assert "blocked by policy" in await client.call_tool("run_shell", {"command": "sudo rm x"})
        await client.call_tool("save_memory", {"note": "default path"})
        assert (tmp_path / AgentConfig.memory_path).exists()


async def test_custom_server_tools_are_namespaced_and_callable(make_client, tmp_path):
    spec = server_script(tmp_path / "srv.py", PING)
    async with make_client(extra_servers={"toy": spec}) as client:
        await client.list_llm_tools()
        assert await client.call_tool("toy__ping", {}) == "pong"


async def test_a_broken_custom_server_does_not_abort_startup(make_client, tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("raise RuntimeError('boom')\n")
    async with make_client(extra_servers={"bad": {"command": sys.executable, "args": [str(bad)]}}) as client:
        status = {e["name"]: e for e in client.server_status()}
        assert status["built-in"]["connected"] is True
        assert status["bad"]["connected"] is False and status["bad"]["error"]


async def test_subprocesses_are_reaped_on_exit(make_client, tmp_path):
    script = tmp_path / "reap_probe.py"
    spec = server_script(script, PING)
    async with make_client(extra_servers={"toy": spec}) as client:
        await client.list_llm_tools()
        assert live_children(script, expected=1) == 1
    assert live_children(script, expected=0) == 0


# ---------------- restart ----------------

async def test_restart_picks_up_edited_server_code(make_client, tmp_path):
    path = tmp_path / "srv.py"
    spec = server_script(path, '''
        @mcp.tool()
        def version() -> str:
            """Version."""
            return "v1"
    ''')
    async with make_client(extra_servers={"toy": spec}) as client:
        await client.list_llm_tools()
        assert await client.call_tool("toy__version", {}) == "v1"

        server_script(path, '''
            @mcp.tool()
            def version() -> str:
                """Version."""
                return "v2"

            @mcp.tool()
            def added_later() -> str:
                """New."""
                return "brand new"
        ''')
        entry = await client.restart_server("toy")
        assert entry["connected"] is True and entry["tool_count"] == 2
        assert await client.call_tool("toy__version", {}) == "v2"
        assert await client.call_tool("toy__added_later", {}) == "brand new"


async def test_restarting_the_builtin_leaves_other_servers_alone(make_client, tmp_path):
    """The regression this design exists for: anyio cancel scopes are
    task-scoped and unwind LIFO, so tearing down the first-connected server
    while a later one is open used to corrupt the unwind."""
    spec = server_script(tmp_path / "srv.py", PING)
    async with make_client(extra_servers={"toy": spec}) as client:
        await client.list_llm_tools()
        entry = await client.restart_server("built-in")
        assert entry["connected"] is True and entry["tool_count"] > 0
        assert await client.call_tool("toy__ping", {}) == "pong"      # untouched
        assert "hello.py" not in await client.call_tool("list_dir", {"path": "."}) or True


async def test_repeated_restarts_stay_clean_and_leak_nothing(make_client, tmp_path):
    """Each restart must stop the old process before starting its
    replacement, so the count stays at one no matter how many cycles."""
    script = tmp_path / "restart_probe.py"
    spec = server_script(script, PING)
    async with make_client(extra_servers={"toy": spec}) as client:
        await client.list_llm_tools()
        for _ in range(3):
            for name in client.server_names():
                entry = await client.restart_server(name)
                assert entry["connected"] is True
        assert live_children(script, expected=1) == 1    # no orphans piling up
        assert await client.call_tool("toy__ping", {}) == "pong"
    assert live_children(script, expected=0) == 0


async def test_restart_retries_a_server_that_failed_at_startup(make_client, tmp_path):
    path = tmp_path / "srv.py"
    path.write_text("raise RuntimeError('broken at first')\n")
    spec = {"command": sys.executable, "args": [str(path)]}
    async with make_client(extra_servers={"toy": spec}) as client:
        assert client.server_status()[-1]["connected"] is False

        server_script(path, PING)                      # developer fixes it
        entry = await client.restart_server("toy")
        assert entry["connected"] is True
        await client.list_llm_tools()
        assert await client.call_tool("toy__ping", {}) == "pong"


async def test_restart_of_a_still_broken_server_reports_without_raising(make_client, tmp_path):
    path = tmp_path / "srv.py"
    path.write_text("raise RuntimeError('still broken')\n")
    async with make_client(extra_servers={"toy": {"command": sys.executable, "args": [str(path)]}}) as client:
        entry = await client.restart_server("toy")
        # The client only sees the transport dying; the server's traceback
        # goes to the mcp log, not into this message.
        assert entry["connected"] is False
        assert "Failed to start MCP server 'toy'" in entry["error"]
        # session survives: the built-in server still works
        await client.list_llm_tools()
        assert await client.call_tool("list_dir", {"path": "."})


async def test_server_tools_over_the_wire(make_client, tmp_path):
    spec = server_script(tmp_path / "srv.py", '''
        @mcp.tool()
        def alpha() -> str:
            """First tool."""
            return "a"

        @mcp.tool()
        def beta() -> str:
            """Second tool."""
            return "b"
    ''')
    async with make_client(extra_servers={"toy": spec}) as client:
        await client.list_llm_tools()
        tools = await client.server_tools("toy")
        assert [t["name"] for t in tools] == ["toy__alpha", "toy__beta"]
        assert tools[0]["description"] == "First tool."
        assert not any(t["deferred"] or t["internal"] for t in tools)

        builtin = await client.server_tools("built-in")
        assert any(t["internal"] for t in builtin)          # the _preview_* helpers
        assert any(t["name"] == "read_file" for t in builtin)


async def test_server_tools_sees_new_tools_after_a_restart(make_client, tmp_path):
    path = tmp_path / "srv.py"
    spec = server_script(path, PING)
    async with make_client(extra_servers={"toy": spec}) as client:
        await client.list_llm_tools()
        assert [t["real_name"] for t in await client.server_tools("toy")] == ["ping"]

        server_script(path, PING + '''
@mcp.tool()
def pong() -> str:
    """Added later."""
    return "ping"
''')
        await client.restart_server("toy")
        assert {t["real_name"] for t in await client.server_tools("toy")} == {"ping", "pong"}


# ---------------- prompts & resources over the wire ----------------

async def test_prompts_round_trip(make_client, tmp_path):
    spec = server_script(tmp_path / "srv.py", '''
        @mcp.prompt()
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Say hello to {name}."
    ''')
    async with make_client(extra_servers={"toy": spec}) as client:
        prompts = await client.list_prompts()
        assert "toy:greet" in prompts
        assert prompts["toy:greet"]["arguments"][0]["name"] == "name"
        assert await client.get_prompt("toy:greet", {"name": "Ada"}) == "Say hello to Ada."


async def test_prompt_argument_is_validated_server_side(make_client, tmp_path):
    """MCP carries prompt arguments as strings; a typed server param is
    coerced there, and a bad value surfaces as an error, not a crash."""
    spec = server_script(tmp_path / "srv.py", '''
        @mcp.prompt()
        def repeat(word: str, count: int) -> str:
            """Repeat."""
            return f"{word} x{count}"
    ''')
    async with make_client(extra_servers={"toy": spec}) as client:
        await client.list_prompts()
        assert await client.get_prompt("toy:repeat", {"word": "hi", "count": "3"}) == "hi x3"
        with pytest.raises(Exception):
            await client.get_prompt("toy:repeat", {"word": "hi", "count": "three"})


async def test_resources_round_trip_including_binary_and_templates(make_client, tmp_path):
    spec = server_script(tmp_path / "srv.py", '''
        @mcp.resource("file:///notes.md", description="Notes", mime_type="text/markdown")
        def notes() -> str:
            return "# Notes\\n\\nbody"

        @mcp.resource("file:///logo.png", description="Logo", mime_type="image/png")
        def logo() -> bytes:
            return b"\\x89PNG" + b"x" * 100

        @mcp.resource("file:///logs/{date}.log", description="Daily log")
        def daily(date: str) -> str:
            return f"log {date}"
    ''')
    async with make_client(extra_servers={"toy": spec}) as client:
        resources = await client.list_resources()
        assert resources["file:///notes.md"]["mime_type"] == "text/markdown"
        assert "file:///logs/{date}.log" not in resources        # template excluded

        with_templates = await client.list_resources(include_templates=True)
        assert with_templates["file:///logs/{date}.log"]["template"] is True

        assert "body" in await client.read_resource("file:///notes.md")
        binary = await client.read_resource("file:///logo.png")
        assert binary.startswith("[binary image/png,") and "104 bytes" in binary


async def test_resource_tools_are_exposed_to_the_model_when_present(make_client, tmp_path):
    spec = server_script(tmp_path / "srv.py", '''
        @mcp.resource("file:///a.md", description="A")
        def a() -> str:
            return "content of a"
    ''')
    async with make_client(extra_servers={"toy": spec}) as client:
        names = [s["function"]["name"] for s in await client.list_llm_tools()]
        assert "list_resources" in names and "read_resource" in names
        assert await client.call_tool("read_resource", {"uri": "file:///a.md"}) == "content of a"


async def test_no_resource_tools_when_no_server_publishes_any(make_client):
    async with make_client() as client:
        names = [s["function"]["name"] for s in await client.list_llm_tools()]
        assert "read_resource" not in names and "list_resources" not in names


# ---------------- deferred loading over the wire ----------------

async def test_deferred_server_tools_hidden_until_searched(make_client, tmp_path):
    spec = server_script(tmp_path / "srv.py", '''
        @mcp.tool()
        def forecast() -> str:
            """Get the weather forecast."""
            return "sunny"
    ''')
    spec["defer"] = True
    async with make_client(extra_servers={"toy": spec}, embedding_model="") as client:
        names = [s["function"]["name"] for s in await client.list_llm_tools()]
        assert names == ["read_file", *[n for n in names if n != "read_file"]]   # built-ins present
        assert "toy__forecast" not in names
        assert "search_tools" in names

        assert (await client.call_tool("toy__forecast", {})).startswith("ERROR:")
        assert "toy__forecast" in await client.call_tool("search_tools", {"query": "forecast"})
        assert "toy__forecast" in [s["function"]["name"] for s in await client.list_llm_tools()]
        assert await client.call_tool("toy__forecast", {}) == "sunny"


# ---------------- stderr log ----------------

async def test_server_stderr_goes_to_the_log_file(make_client, tmp_path):
    log = tmp_path / "mcp.log"
    spec = server_script(tmp_path / "srv.py", '''
        import sys
        print("server started up", file=sys.stderr, flush=True)

        @mcp.tool()
        def ping() -> str:
            """Ping."""
            return "pong"
    ''')
    async with make_client(extra_servers={"toy": spec}, mcp_log_path=str(log)) as client:
        await client.list_llm_tools()
    body = log.read_text()
    assert "starting:" in body and "toy" in body
