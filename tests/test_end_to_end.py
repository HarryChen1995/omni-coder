"""The REPL, the agent and the store wired together, with only the two real
boundaries replaced: the LLM (no model server in a test run) and the MCP
subprocess (spawning one is the `live` suite's job).

Everything between them is the shipping code — `_interactive`'s loop, its
turn dispatch, `_start_turn`/`_turn_finished`, `CodingAgent._run_loop`, the
panes and the SQLite store. These are the paths that only break when the
pieces meet: a line routed to the wrong agent, a message that reaches the
model but never the database, a turn that ends one step before it should.
"""


import pytest

from omni import agent as agent_mod
from omni import cli as cli_mod
from omni.session_store import SessionStore


def assistant(content, tool_calls=None):
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


def calls_read_file(path="notes.txt"):
    return [{"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "%s"}' % path}}]


def user_texts(conversation):
    return [str(m.get("content")) for m in conversation if m["role"] == "user"]


@pytest.fixture
def mcp(mocker):
    """The MCP client, stubbed at the process boundary — one read-only tool."""
    c = mocker.AsyncMock()
    c.list_llm_tools.return_value = [{
        "type": "function",
        "function": {"name": "read_file", "description": "", "parameters": {}},
    }]
    c.list_prompts.return_value = {}
    c.list_resources.return_value = {}
    c.file_exists.return_value = True
    c.call_tool.return_value = "alpha\nbeta\ngamma\n"
    c.server_status = mocker.Mock(return_value=[])
    c.server_names = mocker.Mock(return_value=[])
    c.__aenter__.return_value = c
    c.__aexit__.return_value = False
    mocker.patch.object(cli_mod, "MCPToolClient", return_value=c)
    return c


@pytest.fixture
def llm(mocker):
    """The model, stubbed at the wire. Hands back queued replies and records
    the conversation it was sent each time (deep-copied — the agent mutates
    that list in place)."""
    import copy

    class Model:
        def __init__(self):
            self.replies = []
            self.sent = []
            self.on_call = None

        async def __call__(self, *args, **kwargs):
            self.sent.append(copy.deepcopy(kwargs.get("messages", [])))
            if self.on_call is not None:
                await self.on_call(len(self.sent))
            return self.replies[min(len(self.sent) - 1, len(self.replies) - 1)]

    model = Model()
    mocker.patch.object(agent_mod, "chat", model)
    return model


@pytest.fixture
def repl_driver(mocker, cfg, mcp, llm):
    """Run `_interactive` for real, delivering typed lines on cue.

    Lines are handed over by an async callable rather than a fixed script, so
    a test can deliver one *while* a turn is in flight — which is the whole
    point of the feature under test and something a sequential script cannot
    express."""
    mocker.patch.object(cli_mod, "_make_tui", lambda *a, **k: None)
    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch("omni.ui.final_result")
    mocker.patch("omni.ui.banner")

    async def run(next_line, resume=None, session_name=None):
        async def fake_next(tui, main_pane):
            text = await next_line(main_pane)
            return (None, None) if text is None else (main_pane, text)

        mocker.patch.object(cli_mod, "_next_instruction", fake_next)
        await cli_mod._interactive(cfg, resume, session_name)

    return run


def lines(*script):
    """Deliver each line in order, then end the session."""
    remaining = iter(script)

    async def give(pane):
        return next(remaining, None)

    return give


# ---------------- a line typed while the turn is running ----------------


async def test_a_line_typed_mid_turn_is_answered_by_that_turn(repl_driver, llm, mcp):
    """The headline behaviour, through the real stack: the second line is
    typed while the first turn is reading a file, and that same turn comes
    back having taken it into account."""
    llm.replies = [assistant("Reading it.", calls_read_file()),
                    assistant("Read it, and checked the tests too.")]
    holder = {}

    async def type_while_the_tool_runs(*a, **k):
        # The turn is mid-step, exactly where a person would be typing.
        cli_mod._inject(holder["pane"], "also check the tests", [])
        return "alpha\nbeta\n"

    mcp.call_tool.side_effect = type_while_the_tool_runs

    async def script(pane):
        if "pane" not in holder:
            holder["pane"] = pane
            return "read notes.txt"
        return None

    await repl_driver(script)

    assert len(llm.sent) == 2                       # one turn, two model calls
    assert user_texts(llm.sent[-1]) == ["read notes.txt", "also check the tests"]


async def test_the_whole_exchange_is_stored_as_one_session(repl_driver, llm, mcp, cfg):
    """Reaching the model is only half of it — an injected line that never
    reaches the database would vanish on the next --resume."""
    llm.replies = [assistant("Reading it.", calls_read_file()),
                    assistant("Both done.")]
    holder = {}

    async def type_while_the_tool_runs(*a, **k):
        cli_mod._inject(holder["pane"], "and the tests", [])
        holder["typed"] = True
        return "contents"

    mcp.call_tool.side_effect = type_while_the_tool_runs

    async def script(pane):
        if "pane" not in holder:
            holder["pane"] = pane
            return "read notes.txt"
        return None

    await repl_driver(script)

    store = SessionStore(cfg.db_path)
    sessions = store.list_sessions()
    assert len(sessions) == 1                       # one turn, one session
    stored = store.load_messages(sessions[0]["id"])
    assert user_texts(stored) == ["read notes.txt", "and the tests"]
    # And it sits after the tool result, where a strict server accepts it.
    assert [m["role"] for m in stored] == [
        "system", "user", "assistant", "tool", "user", "assistant"]


async def test_a_turn_with_nothing_typed_into_it_is_unchanged(repl_driver, llm, mcp, cfg):
    """The ordinary path has to be exactly what it was."""
    llm.replies = [assistant("Reading it.", calls_read_file()), assistant("Done.")]

    await repl_driver(lines("read notes.txt"))

    assert user_texts(llm.sent[-1]) == ["read notes.txt"]
    store = SessionStore(cfg.db_path)
    stored = store.load_messages(store.list_sessions()[0]["id"])
    assert [m["role"] for m in stored] == ["system", "user", "assistant", "tool", "assistant"]


async def test_two_lines_typed_in_sequence_are_two_turns_in_one_session(repl_driver, llm, cfg):
    """Nothing about injection may change what typing at an idle prompt
    does: each line is its own turn, both continuing the same session."""
    llm.replies = [assistant("First answer."), assistant("Second answer.")]

    await repl_driver(lines("do the first thing", "now the second"))

    store = SessionStore(cfg.db_path)
    assert len(store.list_sessions()) == 1
    stored = store.load_messages(store.list_sessions()[0]["id"])
    assert user_texts(stored) == ["do the first thing", "now the second"]


# ---------------- resuming by name, end to end ----------------


async def test_a_session_can_be_resumed_by_name_and_is_labelled_by_it(
        repl_driver, llm, cfg, mocker):
    """Name it on one run, resume it by that name on the next, and the
    terminal tab says what it is called rather than its id."""
    llm.replies = [assistant("Started.")]
    await repl_driver(lines("begin the refactor"), session_name="my refactor")

    store = SessionStore(cfg.db_path)
    sid = store.list_sessions()[0]["id"]

    llm.replies = [assistant("Continued.")]
    title = mocker.patch("omni.ui.set_terminal_title")
    await repl_driver(lines("carry on"), resume="  MY REFACTOR ")

    # Same session, continued — not a second one started alongside it.
    assert len(store.list_sessions()) == 1
    assert user_texts(store.load_messages(sid)) == ["begin the refactor", "carry on"]
    assert {call.args[0] for call in title.call_args_list} == {"my refactor"}


async def test_the_replayed_conversation_carries_an_injected_line(
        repl_driver, llm, mcp, cfg, mocker):
    """A resumed session has to read back exactly as it happened, mid-turn
    interjection included."""
    llm.replies = [assistant("Reading.", calls_read_file()), assistant("Done.")]
    holder = {}

    async def type_while_the_tool_runs(*a, **k):
        cli_mod._inject(holder["pane"], "and the tests", [])
        return "contents"

    mcp.call_tool.side_effect = type_while_the_tool_runs

    async def script(pane):
        if "pane" not in holder:
            holder["pane"] = pane
            return "read notes.txt"
        return None

    await repl_driver(script, session_name="threaded")

    llm.replies = [assistant("Still here.")]
    await repl_driver(lines("what did I ask you?"), resume="threaded")

    # Turn two is sent the whole earlier conversation, interjection included.
    assert user_texts(llm.sent[-1]) == [
        "read notes.txt", "and the tests", "what did I ask you?"]
