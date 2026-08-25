"""CodingAgent.run / _run_loop / _call_model / compact_history.

The MCP client and `chat` are mocked, so the loop's control flow — tool
dispatch, parallel vs sequential execution, approval denial, session
persistence, cancellation, retries — is driven deterministically.
"""

import asyncio
import copy

import pytest

from omni import agent as agent_mod
from omni.agent import CodingAgent
from omni.llm_client import LLMError


def text_reply(content):
    return {"role": "assistant", "content": content}


def tool_reply(*calls):
    """An assistant turn requesting tool calls. Each call is (name, args_json)."""
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": a}}
        for i, (n, a) in enumerate(calls)]}


@pytest.fixture
def client(mocker):
    """A stand-in MCPToolClient: two safe tools and one write tool."""
    c = mocker.AsyncMock()
    c.list_llm_tools.return_value = [
        {"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
        for n in ("read_file", "list_dir", "write_file", "run_shell")
    ]
    c.call_tool.return_value = "tool output"
    c.file_exists.return_value = True
    return c


@pytest.fixture
def agent(cfg, mocker):
    mocker.patch.object(agent_mod.ui, "step_display")
    mocker.patch.object(agent_mod.ui, "elapsed_note")
    mocker.patch.object(agent_mod.ui, "banner")
    return CodingAgent(cfg)


def replies(mocker, *messages):
    """Patch chat() to return each message in turn.

    The mock also snapshots the `messages` list on every call (as
    `.sent[i]`): the agent mutates that list in place, so a mock's recorded
    call args — which hold a reference, not a copy — would otherwise all
    show the final state."""
    async def fake(*args, **kwargs):
        fake.sent.append(copy.deepcopy(kwargs.get("messages", [])))
        item = messages[min(fake.calls, len(messages) - 1)]
        fake.calls += 1
        if isinstance(item, BaseException):
            raise item
        return item

    fake.calls = 0
    fake.sent = []
    m = mocker.patch.object(agent_mod, "chat", side_effect=fake)
    m.sent = fake.sent
    return m


# ---------------- _call_model ----------------

async def test_call_model_returns_the_message(agent, mocker):
    replies(mocker, text_reply("done"))
    assert await agent._call_model([], []) == text_reply("done")


async def test_call_model_passes_config_through(agent, mocker):
    agent.cfg.llm_host, agent.cfg.llm_api_key = "http://h", "k"
    agent.cfg.llm_timeout_s = 55.0
    m = replies(mocker, text_reply("x"))
    schemas = [{"type": "function"}]
    await agent._call_model([{"role": "user", "content": "q"}], schemas)
    kwargs = m.await_args.kwargs
    assert kwargs["model"] == agent.cfg.model and kwargs["tools"] == schemas
    assert kwargs["base_url"] == "http://h" and kwargs["api_key"] == "k"
    assert kwargs["timeout"] == 55.0


async def test_call_model_retries_then_succeeds(agent, mocker):
    mocker.patch.object(agent_mod.asyncio, "sleep", mocker.AsyncMock())
    m = replies(mocker, LLMError("503"), text_reply("recovered"))
    assert (await agent._call_model([], []))["content"] == "recovered"
    assert m.await_count == 2


async def test_call_model_gives_up_after_max_retries(agent, mocker):
    mocker.patch.object(agent_mod.asyncio, "sleep", mocker.AsyncMock())
    agent.cfg.max_retries = 2
    m = replies(mocker, LLMError("down"))
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        await agent._call_model([], [])
    assert m.await_count == 2


async def test_call_model_retries_unexpected_errors_too(agent, mocker):
    mocker.patch.object(agent_mod.asyncio, "sleep", mocker.AsyncMock())
    replies(mocker, ValueError("weird"), text_reply("ok"))
    assert (await agent._call_model([], []))["content"] == "ok"


# ---------------- run(): happy path & session bookkeeping ----------------

async def test_run_returns_final_text_and_marks_session_done(agent, client, mocker):
    replies(mocker, text_reply("all finished"))
    assert await agent.run("do a thing", client=client) == "all finished"
    row = agent.store.list_sessions()[0]
    assert row["status"] == "done" and row["summary"] == "all finished"
    assert row["task"] == "do a thing"


async def test_run_seeds_system_prompt_and_task(agent, client, mocker):
    m = replies(mocker, text_reply("fin"))
    await agent.run("my task", client=client)
    sent = m.sent[0]
    assert sent[0]["role"] == "system" and agent_mod.SYSTEM_PROMPT in sent[0]["content"]
    assert sent[1] == {"role": "user", "content": "my task"}


async def test_run_folds_in_project_memory(agent, client, mocker, project_root):
    (project_root / "agent_memory.md").write_text("- prefers tabs")
    m = replies(mocker, text_reply("fin"))
    await agent.run("t", client=client)
    assert "prefers tabs" in m.sent[0][0]["content"]


async def test_run_persists_every_message(agent, client, mocker):
    replies(mocker, tool_reply(("read_file", '{"path": "x"}')), text_reply("done"))
    await agent.run("t", client=client)
    roles = [m["role"] for m in agent.store.load_messages(agent.session_id)]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


async def test_run_sets_session_id(agent, client, mocker):
    replies(mocker, text_reply("x"))
    await agent.run("t", client=client)
    assert agent.session_id and agent.store.session_exists(agent.session_id)


async def test_run_names_the_session(agent, client, mocker):
    replies(mocker, text_reply("x"))
    await agent.run("t", client=client, session_name="my-run")
    assert agent.store.resolve_session_id("my-run") == agent.session_id


# ---------------- run(): resume ----------------

async def test_resume_loads_prior_history(agent, client, mocker):
    replies(mocker, text_reply("first"))
    await agent.run("original", client=client)
    sid = agent.session_id

    m = replies(mocker, text_reply("second"))
    await agent.run("follow up", resume_session_id=sid, client=client)
    sent = m.sent[0]
    assert sent[1]["content"] == "original"          # prior turn carried over
    assert sent[-1] == {"role": "user", "content": "follow up"}


async def test_resume_by_name(agent, client, mocker):
    replies(mocker, text_reply("x"))
    await agent.run("t", client=client, session_name="named")
    replies(mocker, text_reply("y"))
    assert await agent.run("more", resume_session_id="named", client=client) == "y"


async def test_resume_unknown_session_raises(agent, client, mocker):
    replies(mocker, text_reply("x"))
    with pytest.raises(ValueError, match="No session found"):
        await agent.run("t", resume_session_id="ghost", client=client)


async def test_resume_without_new_task_does_not_append_a_user_message(agent, client, mocker):
    replies(mocker, text_reply("a"))
    await agent.run("orig", client=client)
    m = replies(mocker, text_reply("b"))
    await agent.run("", resume_session_id=agent.session_id, client=client)
    assert m.sent[0][-1]["role"] != "user"


async def test_resume_does_not_rewrite_existing_history(agent, client, mocker):
    replies(mocker, text_reply("a"))
    await agent.run("orig", client=client)
    before = len(agent.store.load_messages(agent.session_id))
    replies(mocker, text_reply("b"))
    await agent.run("next", resume_session_id=agent.session_id, client=client)
    after = agent.store.load_messages(agent.session_id)
    assert len(after) == before + 2          # one user + one assistant appended
    assert after[1]["content"] == "orig"     # original entries intact


# ---------------- tool execution ----------------

async def test_tool_call_is_dispatched_and_result_fed_back(agent, client, mocker):
    client.call_tool.return_value = "file contents here"
    m = replies(mocker, tool_reply(("read_file", '{"path": "a.py"}')), text_reply("done"))
    await agent.run("t", client=client)
    client.call_tool.assert_any_await("read_file", {"path": "a.py"})
    assert any(x["role"] == "tool" and "file contents here" in x["content"] for x in m.sent[-1])


async def test_malformed_arguments_reported_without_calling_the_tool(agent, client, mocker):
    replies(mocker, tool_reply(("read_file", "{not json")), text_reply("done"))
    await agent.run("t", client=client)
    client.call_tool.assert_not_awaited()
    tool_msg = [m for m in agent.store.load_messages(agent.session_id) if m["role"] == "tool"][0]
    assert "malformed arguments" in tool_msg["content"]


async def test_all_safe_tools_run_concurrently(agent, client, mocker):
    """A step of read-only calls is parallelised."""
    running = []

    async def slow(name, args):
        running.append(name)
        await asyncio.sleep(0.02)
        return f"{name} done"

    client.call_tool.side_effect = slow
    replies(mocker, tool_reply(("read_file", "{}"), ("list_dir", "{}")), text_reply("fin"))
    await agent.run("t", client=client)
    assert running == ["read_file", "list_dir"]   # both started before either finished


async def test_step_with_a_write_tool_runs_sequentially(agent, client, mocker):
    """Ordering matters once state can change, so the whole step serialises."""
    order = []

    async def track(name, args):
        order.append(f"start:{name}")
        await asyncio.sleep(0.01)
        order.append(f"end:{name}")
        return "ok"

    agent.cfg.auto_approve = True
    client.call_tool.side_effect = track
    replies(mocker, tool_reply(("write_file", '{"path":"a"}'), ("read_file", "{}")), text_reply("fin"))
    await agent.run("t", client=client)
    assert order == ["start:write_file", "end:write_file", "start:read_file", "end:read_file"]


async def test_denied_tool_is_not_executed(agent, client, mocker):
    mocker.patch.object(agent_mod.ui, "request_approval", mocker.AsyncMock(return_value=False))
    replies(mocker, tool_reply(("write_file", '{"path": "x"}')), text_reply("understood"))
    await agent.run("t", client=client)
    client.call_tool.assert_not_awaited()
    tool_msg = [m for m in agent.store.load_messages(agent.session_id) if m["role"] == "tool"][0]
    assert "Denied by human reviewer" in tool_msg["content"]


async def test_tool_exception_is_reported_to_the_model(agent, client, mocker):
    client.call_tool.side_effect = RuntimeError("tool exploded")
    replies(mocker, tool_reply(("read_file", "{}")), text_reply("noted"))
    await agent.run("t", client=client)
    tool_msg = [m for m in agent.store.load_messages(agent.session_id) if m["role"] == "tool"][0]
    assert "ERROR" in tool_msg["content"] and "tool exploded" in tool_msg["content"]


async def test_search_tools_result_refreshes_the_schemas(agent, client, mocker):
    client.call_tool.return_value = "Loaded 1 tool(s)"
    replies(mocker, tool_reply(("search_tools", '{"query": "x"}')), text_reply("fin"))
    await agent.run("t", client=client)
    assert client.list_llm_tools.await_count >= 2   # re-listed after the reveal


async def test_plain_text_tool_call_is_recovered(agent, client, mocker):
    """A model that prints the call as text instead of using the API must not
    silently end the run."""
    replies(mocker,
            text_reply('{"name": "read_file", "arguments": {"path": "a.py"}}'),
            text_reply("done"))
    await agent.run("t", client=client)
    client.call_tool.assert_any_await("read_file", {"path": "a.py"})


# ---------------- intent parsing ----------------

async def test_intent_block_is_injected_when_enabled(agent, client, mocker):
    agent.cfg.parse_intent = True
    from omni.intent import Intent
    mocker.patch.object(agent_mod, "extract_intent",
                        mocker.AsyncMock(return_value=Intent(task_type="bugfix", summary="s")))
    mocker.patch.object(agent_mod.ui, "intent_panel")
    m = replies(mocker, text_reply("fin"))
    await agent.run("fix it", client=client)
    assert any("Parsed intent" in str(x.get("content")) for x in m.sent[0])


async def test_high_risk_intent_forces_approval(agent, client, mocker):
    agent.cfg.parse_intent = True
    agent.cfg.auto_approve = True
    from omni.intent import Intent
    mocker.patch.object(agent_mod, "extract_intent",
                        mocker.AsyncMock(return_value=Intent(risk_level="high")))
    mocker.patch.object(agent_mod.ui, "intent_panel")
    mocker.patch.object(agent_mod.ui, "high_risk_warning")
    approval = mocker.patch.object(agent_mod.ui, "request_approval", mocker.AsyncMock(return_value=False))
    replies(mocker, tool_reply(("write_file", '{"path":"x"}')), text_reply("ok"))
    await agent.run("delete everything", client=client)
    assert agent.force_approval is True
    approval.assert_awaited()          # prompted despite --auto-approve


async def test_intent_is_parsed_for_every_new_instruction(agent, client, mocker):
    """Including later turns of an interactive session, which all arrive as
    resumes — that's where high-risk detection used to go dark."""
    agent.cfg.parse_intent = True
    from omni.intent import Intent
    extract = mocker.patch.object(agent_mod, "extract_intent",
                                  mocker.AsyncMock(return_value=Intent()))
    mocker.patch.object(agent_mod.ui, "intent_panel")
    replies(mocker, text_reply("a"))
    await agent.run("first", client=client)
    assert extract.await_count == 1
    replies(mocker, text_reply("b"))
    await agent.run("second", resume_session_id=agent.session_id, client=client)
    assert extract.await_count == 2


async def test_intent_parsing_is_skipped_when_resuming_with_no_new_task(agent, client, mocker):
    agent.cfg.parse_intent = True
    from omni.intent import Intent
    extract = mocker.patch.object(agent_mod, "extract_intent",
                                  mocker.AsyncMock(return_value=Intent()))
    mocker.patch.object(agent_mod.ui, "intent_panel")
    replies(mocker, text_reply("a"))
    await agent.run("first", client=client)
    replies(mocker, text_reply("b"))
    await agent.run("", resume_session_id=agent.session_id, client=client)
    assert extract.await_count == 1    # nothing new to parse


# ---------------- budget / compaction / limits ----------------

async def test_history_is_compacted_when_over_budget(agent, client, mocker):
    agent.cfg.context_char_budget = 10
    compact = mocker.patch.object(agent_mod, "_compact_messages",
                                  mocker.AsyncMock(side_effect=lambda m, *a: m))
    replies(mocker, text_reply("fin"))
    await agent.run("a task long enough to exceed the tiny budget", client=client)
    compact.assert_awaited()


async def test_history_is_not_compacted_under_budget(agent, client, mocker):
    agent.cfg.context_char_budget = 10_000_000
    compact = mocker.patch.object(agent_mod, "_compact_messages", mocker.AsyncMock())
    replies(mocker, text_reply("fin"))
    await agent.run("t", client=client)
    compact.assert_not_awaited()


async def test_max_steps_terminates_the_loop(agent, client, mocker):
    agent.cfg.max_steps = 3
    replies(mocker, tool_reply(("read_file", "{}")))   # never returns a final answer
    out = await agent.run("t", client=client)
    assert "Max steps reached" in out
    assert agent.store.list_sessions()[0]["status"] == "max_steps"


# ---------------- failure & cancellation ----------------

async def test_unexpected_failure_marks_session_error_and_reraises(agent, client, mocker):
    mocker.patch.object(agent_mod.asyncio, "sleep", mocker.AsyncMock())
    agent.cfg.max_retries = 1
    replies(mocker, LLMError("gone"))
    with pytest.raises(RuntimeError):
        await agent.run("t", client=client)
    row = agent.store.list_sessions()[0]
    assert row["status"] == "error"


async def test_cancellation_marks_session_interrupted(agent, client, mocker):
    """Ctrl+C during a turn: CancelledError derives from BaseException, so it
    needs its own handler or the session would stay 'running' forever."""
    async def hang(*a, **k):
        await asyncio.sleep(60)

    mocker.patch.object(CodingAgent, "_run_loop", hang)
    task = asyncio.ensure_future(agent.run("t", client=client))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    row = agent.store.list_sessions()[0]
    assert row["status"] == "interrupted" and "Ctrl+C" in row["summary"]


# ---------------- compact_history (/compact) ----------------

async def test_compact_history_persists_the_shrunk_history(agent, mocker):
    agent.cfg.compact_keep_last = 2
    sid = agent.store.create_session("/p", "m", "t")
    agent.store.append_message(sid, 0, {"role": "system", "content": "sys"})
    agent.store.append_message(sid, 1, {"role": "user", "content": "the task"})
    for i in range(18):
        agent.store.append_message(sid, i + 2, {"role": "user", "content": f"m{i}"})
    mocker.patch.object(agent_mod, "chat", mocker.AsyncMock(return_value={"content": "BRIEFING"}))
    mocker.patch.object(agent_mod.ui, "compacted")

    out = await agent.compact_history(sid)
    assert "Compacted 20 messages down to 5" in out
    reloaded = agent.store.load_messages(sid)
    assert len(reloaded) == 5 and "BRIEFING" in reloaded[2]["content"]
    # system prompt + the task itself are the protected head
    assert [m["content"] for m in reloaded[:2]] == ["sys", "the task"]


async def test_compact_history_noop_for_short_history(agent, mocker):
    sid = agent.store.create_session("/p", "m", "t")
    agent.store.append_message(sid, 0, {"role": "user", "content": "x"})
    assert "Nothing to compact" in await agent.compact_history(sid)
    assert len(agent.store.load_messages(sid)) == 1


# ---------------- tool_call_id pairing ----------------

def tool_reply_without_ids(*names):
    """Some OpenAI-compatible servers hand back tool calls with no id."""
    return {"role": "assistant", "content": None, "tool_calls": [
        {"type": "function", "function": {"name": n, "arguments": "{}"}} for n in names]}


def sent_pairs(snapshot):
    """(assistant tool_call ids, tool-result tool_call_ids) from one snapshot
    of the messages list as it was handed to chat()."""
    call_ids = [c["id"] for m in snapshot if m.get("tool_calls") for c in m["tool_calls"]]
    result_ids = [m.get("tool_call_id") for m in snapshot if m["role"] == "tool"]
    return call_ids, result_ids


async def test_tool_results_name_the_call_they_answer(agent, client, mocker):
    """role="tool" without tool_call_id is rejected outright by a strict
    OpenAI-compatible server (vLLM, OpenAI, OpenRouter) — Ollama's tolerance
    was the only reason multi-step runs worked."""
    m = replies(mocker, tool_reply(("read_file", '{"path": "a"}')), text_reply("done"))
    await agent.run("t", client=client)
    call_ids, result_ids = sent_pairs(m.sent[1])
    assert call_ids == result_ids == ["c0"]


async def test_every_parallel_call_gets_its_own_pairing(agent, client, mocker):
    m = replies(mocker,
                tool_reply(("read_file", '{"path": "a"}'), ("list_dir", '{"path": "."}')),
                text_reply("done"))
    await agent.run("t", client=client)
    call_ids, result_ids = sent_pairs(m.sent[1])
    assert call_ids == result_ids == ["c0", "c1"]


async def test_ids_are_synthesized_when_the_server_omits_them(agent, client, mocker):
    m = replies(mocker, tool_reply_without_ids("read_file", "list_dir"), text_reply("done"))
    await agent.run("t", client=client)
    call_ids, result_ids = sent_pairs(m.sent[1])
    assert call_ids == result_ids == ["call_1_0", "call_1_1"]


async def test_a_malformed_call_still_gets_a_paired_result(agent, client, mocker):
    """The error report goes back as a tool message like any other result, so
    it needs the id too or the whole request is rejected."""
    m = replies(mocker, tool_reply(("read_file", "{not json")), text_reply("done"))
    await agent.run("t", client=client)
    call_ids, result_ids = sent_pairs(m.sent[1])
    assert call_ids == result_ids == ["c0"]
    assert "malformed arguments" in [x for x in m.sent[1] if x["role"] == "tool"][0]["content"]


async def test_recovered_text_tool_calls_are_paired_too(agent, client, mocker):
    m = replies(mocker,
                text_reply('{"name": "read_file", "arguments": {"path": "a"}}'),
                text_reply("done"))
    await agent.run("t", client=client)
    call_ids, result_ids = sent_pairs(m.sent[1])
    assert call_ids == result_ids == ["fallback_0"]


async def test_tool_call_ids_survive_a_resume(agent, client, mocker):
    replies(mocker, tool_reply(("read_file", '{"path": "a"}')), text_reply("a"))
    await agent.run("first", client=client)
    m = replies(mocker, text_reply("b"))
    await agent.run("second", resume_session_id=agent.session_id, client=client)
    call_ids, result_ids = sent_pairs(m.sent[0])
    assert call_ids == result_ids == ["c0"]


# ---------------- per-turn intent / force_approval ----------------

async def test_force_approval_is_re_evaluated_every_turn(agent, client, mocker):
    """Latched for the process, one high-risk turn made every later turn
    prompt — and a low-risk turn 1 left a high-risk turn 5 ungated."""
    from omni.intent import Intent
    agent.cfg.parse_intent = True
    mocker.patch.object(agent_mod.ui, "intent_panel")
    mocker.patch.object(agent_mod.ui, "high_risk_warning")
    mocker.patch.object(agent_mod, "extract_intent", mocker.AsyncMock(
        side_effect=[Intent(risk_level="high"), Intent(risk_level="low")]))

    replies(mocker, text_reply("a"))
    await agent.run("drop the tables", client=client)
    assert agent.force_approval is True

    replies(mocker, text_reply("b"))
    await agent.run("add a docstring", resume_session_id=agent.session_id, client=client)
    assert agent.force_approval is False


async def test_intent_block_is_inserted_before_the_new_instruction(agent, client, mocker):
    from omni.intent import Intent
    agent.cfg.parse_intent = True
    mocker.patch.object(agent_mod.ui, "intent_panel")
    mocker.patch.object(agent_mod, "extract_intent",
                        mocker.AsyncMock(return_value=Intent(summary="parsed")))
    replies(mocker, text_reply("a"))
    await agent.run("first", client=client)
    replies(mocker, text_reply("b"))
    await agent.run("second", resume_session_id=agent.session_id, client=client)

    stored = agent.store.load_messages(agent.session_id)
    # Turn 1's messages keep their positions; the new block lands directly
    # before the instruction it describes rather than at index 1, ahead of
    # history that was already written out.
    assert [m["role"] for m in stored[:4]] == ["system", "system", "user", "assistant"]
    at = next(i for i, m in enumerate(stored) if m["content"] == "second")
    assert stored[at]["role"] == "user"
    assert stored[at - 1]["role"] == "system" and "Parsed intent" in stored[at - 1]["content"]


# ---------------- automatic compaction is persisted ----------------

async def test_automatic_compaction_is_written_back_to_the_store(agent, client, mocker):
    """Otherwise the DB keeps the full pre-compaction history and resuming
    reloads everything that was just summarized away."""
    agent.cfg.context_char_budget = 10
    mocker.patch.object(agent_mod, "_compact_messages", mocker.AsyncMock(
        side_effect=lambda msgs, *a: [msgs[0], {"role": "system", "content": "BRIEFING"}]))
    replies(mocker, text_reply("fin"))
    await agent.run("a task long enough to exceed the tiny budget", client=client)

    stored = agent.store.load_messages(agent.session_id)
    assert [m["content"] for m in stored[1:]] == ["BRIEFING", "fin"]
    assert len(stored) == 3   # renumbered from scratch, not appended after the old rows


async def test_a_no_op_compaction_leaves_the_store_alone(agent, client, mocker):
    agent.cfg.context_char_budget = 10
    mocker.patch.object(agent_mod, "_compact_messages",
                        mocker.AsyncMock(side_effect=lambda msgs, *a: msgs))
    replace = mocker.spy(agent.store, "replace_messages")
    replies(mocker, text_reply("fin"))
    await agent.run("a task long enough to exceed the tiny budget", client=client)
    replace.assert_not_called()


# ---------------- reasoning ----------------

def reasoning_reply(content, reasoning, key="reasoning_content"):
    return {"role": "assistant", "content": content, key: reasoning}


async def test_reasoning_is_shown_collapsed_beside_an_answer(agent, client, mocker):
    """The normal shape for a reasoning model: an answer plus a much longer
    chain of thought, which would bury the answer if printed in full."""
    note = mocker.patch.object(agent_mod.ui, "reasoning_note")
    replies(mocker, reasoning_reply("The answer.", "Long chain of thought."))
    await agent.run("t", client=client)
    note.assert_called_once_with("Long chain of thought.")
    assert agent.last_reasoning == "Long chain of thought."


async def test_plain_reasoning_key_is_shown_too(agent, client, mocker):
    note = mocker.patch.object(agent_mod.ui, "reasoning_note")
    replies(mocker, reasoning_reply("A.", "Thinking.", key="reasoning"))
    await agent.run("t", client=client)
    note.assert_called_once_with("Thinking.")


async def test_reasoning_that_is_the_answer_is_not_echoed(agent, client, mocker):
    """llm_client falls back to the reasoning field when a reply carries
    nothing else; printing the note as well would show it twice."""
    note = mocker.patch.object(agent_mod.ui, "reasoning_note")
    replies(mocker, reasoning_reply("Only thinking.", "Only thinking."))
    await agent.run("t", client=client)
    note.assert_not_called()
    assert agent.last_reasoning == ""


async def test_no_reasoning_field_prints_nothing(agent, client, mocker):
    note = mocker.patch.object(agent_mod.ui, "reasoning_note")
    replies(mocker, text_reply("just an answer"))
    await agent.run("t", client=client)
    note.assert_not_called()


async def test_reasoning_on_a_tool_calling_turn_is_shown_above_the_tools(agent, client, mocker):
    note = mocker.patch.object(agent_mod.ui, "reasoning_note")
    msg = tool_reply(("read_file", '{"path": "a"}'))
    msg["content"] = "Reading it."
    msg["reasoning_content"] = "I should read the file first."
    replies(mocker, msg, text_reply("done"))
    await agent.run("t", client=client)
    note.assert_called_once_with("I should read the file first.")
