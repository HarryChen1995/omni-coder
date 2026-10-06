"""CodingAgent.run / _run_loop / _call_model / compact_history.

The MCP client and `chat` are mocked, so the loop's control flow — tool
dispatch, parallel vs sequential execution, approval denial, session
persistence, cancellation, retries — is driven deterministically.
"""

import asyncio
import copy
import time

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


# ---------------- budget / compaction / limits ----------------

async def test_history_is_compacted_when_over_budget(agent, client, mocker):
    agent.cfg.context_window_budget = 10
    compact = mocker.patch.object(agent_mod, "_compact_messages",
                                  mocker.AsyncMock(side_effect=lambda m, *a, **kw: m))
    replies(mocker, text_reply("fin"))
    await agent.run("a task long enough to exceed the tiny budget", client=client)
    compact.assert_awaited()


async def test_history_is_not_compacted_under_budget(agent, client, mocker):
    agent.cfg.context_window_budget = 10_000_000
    compact = mocker.patch.object(agent_mod, "_compact_messages", mocker.AsyncMock())
    replies(mocker, text_reply("fin"))
    await agent.run("t", client=client)
    compact.assert_not_awaited()


async def test_compaction_triggers_on_reported_tokens_not_char_count(agent, client, mocker):
    """The budget is tokens: once the server reports a prompt larger than it,
    compaction fires even though the char count of a tiny task is nowhere
    near — it's the server's prompt_tokens that's watched, not characters."""
    agent.cfg.context_window_budget = 100
    agent._last_prompt_tokens = 5_000           # the server says the context is large
    compact = mocker.patch.object(agent_mod, "_compact_messages",
                                  mocker.AsyncMock(side_effect=lambda m, *a, **kw: m))
    replies(mocker, text_reply("fin"))
    await agent.run("hi", client=client)         # only a few characters
    compact.assert_awaited()


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


# ---------------- automatic compaction is persisted ----------------

async def test_automatic_compaction_is_written_back_to_the_store(agent, client, mocker):
    """Otherwise the DB keeps the full pre-compaction history and resuming
    reloads everything that was just summarized away."""
    agent.cfg.context_window_budget = 10
    mocker.patch.object(agent_mod, "_compact_messages", mocker.AsyncMock(
        side_effect=lambda msgs, *a, **kw: [msgs[0], {"role": "system", "content": "BRIEFING"}]))
    replies(mocker, text_reply("fin"))
    await agent.run("a task long enough to exceed the tiny budget", client=client)

    stored = agent.store.load_messages(agent.session_id)
    assert [m["content"] for m in stored[1:]] == ["BRIEFING", "fin"]
    assert len(stored) == 3   # renumbered from scratch, not appended after the old rows


async def test_a_no_op_compaction_leaves_the_store_alone(agent, client, mocker):
    agent.cfg.context_window_budget = 10
    mocker.patch.object(agent_mod, "_compact_messages",
                        mocker.AsyncMock(side_effect=lambda msgs, *a, **kw: msgs))
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
    note.assert_called_once_with("Long chain of thought.", 1)
    assert agent.last_reasoning == "Long chain of thought."
    assert agent.reasoning_log == ["Long chain of thought."]


async def test_plain_reasoning_key_is_shown_too(agent, client, mocker):
    note = mocker.patch.object(agent_mod.ui, "reasoning_note")
    replies(mocker, reasoning_reply("A.", "Thinking.", key="reasoning"))
    await agent.run("t", client=client)
    note.assert_called_once_with("Thinking.", 1)


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
    note.assert_called_once_with("I should read the file first.", 1)


# ---------------- system prompt ----------------

def system_of(sent):
    return next(m["content"] for m in sent if m["role"] == "system")


async def test_built_in_system_prompt_is_the_default(agent, client, mocker):
    m = replies(mocker, text_reply("ok"))
    await agent.run("t", client=client)
    assert system_of(m.sent[0]) == agent_mod.SYSTEM_PROMPT


async def test_configured_system_prompt_replaces_the_built_in(agent, client, mocker):
    agent.cfg.system_prompt = "You are a terse reviewer. Answer in one line."
    m = replies(mocker, text_reply("ok"))
    await agent.run("t", client=client)
    sent = system_of(m.sent[0])
    assert sent == "You are a terse reviewer. Answer in one line."
    assert "edit_file" not in sent          # the built-in one is gone, not merged


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
async def test_blank_system_prompt_falls_back_to_the_built_in(agent, client, mocker, blank):
    agent.cfg.system_prompt = blank
    m = replies(mocker, text_reply("ok"))
    await agent.run("t", client=client)
    assert system_of(m.sent[0]) == agent_mod.SYSTEM_PROMPT


async def test_project_memory_still_rides_on_a_custom_prompt(agent, client, mocker, project_root):
    (project_root / "agent_memory.md").write_text("- [2026-01-01] uses pytest\n")
    agent.cfg.system_prompt = "Custom."
    m = replies(mocker, text_reply("ok"))
    await agent.run("t", client=client)
    sent = system_of(m.sent[0])
    assert sent.startswith("Custom.") and "uses pytest" in sent


async def test_a_resumed_session_keeps_the_prompt_it_started_with(agent, client, mocker):
    """The prompt is stored as the session's first message, so changing the
    flag later must not rewrite the history of a session already underway."""
    agent.cfg.system_prompt = "Original prompt."
    replies(mocker, text_reply("a"))
    await agent.run("first", client=client)

    agent.cfg.system_prompt = "Different prompt."
    m = replies(mocker, text_reply("b"))
    await agent.run("second", resume_session_id=agent.session_id, client=client)
    assert system_of(m.sent[0]) == "Original prompt."


async def test_a_session_can_be_resumed_by_the_name_it_was_given(agent, client, mocker):
    """The whole point of --session-name: getting back in without the id."""
    replies(mocker, text_reply("a"))
    await agent.run("first", client=client, session_name="my refactor")
    started = agent.session_id

    replies(mocker, text_reply("b"))
    await agent.run("second", resume_session_id="my refactor", client=client)
    assert agent.session_id == started


async def test_resuming_by_name_ignores_case_and_surrounding_space(agent, client, mocker):
    replies(mocker, text_reply("a"))
    await agent.run("first", client=client, session_name="My Refactor")
    started = agent.session_id

    replies(mocker, text_reply("b"))
    await agent.run("second", resume_session_id="  my refactor ", client=client)
    assert agent.session_id == started


async def test_a_name_that_matches_nothing_still_says_so(agent, client, mocker):
    replies(mocker, text_reply("a"))
    with pytest.raises(ValueError, match="No session found"):
        await agent.run("t", resume_session_id="never-existed", client=client)


async def test_the_resume_banner_calls_the_session_by_its_name(agent, client, mocker):
    """The id says nothing you can read; the name is what was typed to get
    back here."""
    replies(mocker, text_reply("a"))
    await agent.run("first", client=client, session_name="my refactor")

    banner = mocker.patch("omni.ui.banner")
    replies(mocker, text_reply("b"))
    await agent.run("second", resume_session_id="my refactor", client=client, show_banner=True)
    assert "my refactor" in banner.call_args.args[0]


async def test_an_unnamed_session_is_still_announced_by_its_id(agent, client, mocker):
    replies(mocker, text_reply("a"))
    await agent.run("first", client=client)
    started = agent.session_id

    banner = mocker.patch("omni.ui.banner")
    replies(mocker, text_reply("b"))
    await agent.run("second", resume_session_id=started, client=client, show_banner=True)
    assert started in banner.call_args.args[0]


# ---------------- token accounting ----------------
#
# The count beside the spinner should be what the session actually cost, so
# everything that spends tokens on the agent's behalf has to be folded in —
# not just the replies you can see.

def usage_reply(content, prompt=0, completion=0):
    """A chat() stand-in that reports usage the way a server does."""
    async def fake(*args, **kwargs):
        if kwargs.get("usage") is not None:
            kwargs["usage"].update({"prompt_tokens": prompt, "completion_tokens": completion})
        return {"role": "assistant", "content": content}
    return fake


async def test_a_turn_counts_its_own_calls(agent, client, mocker):
    mocker.patch.object(agent_mod, "chat", side_effect=usage_reply("done", 1200, 60))
    await agent.run("t", client=client)
    assert agent.tokens == {"prompt": 1200, "completion": 60}


async def test_counts_accumulate_across_turns(agent, client, mocker):
    mocker.patch.object(agent_mod, "chat", side_effect=usage_reply("done", 100, 10))
    await agent.run("first", client=client)
    await agent.run("second", resume_session_id=agent.session_id, client=client)
    assert agent.tokens == {"prompt": 200, "completion": 20}


async def test_compaction_is_counted(agent, client, mocker):
    agent.cfg.context_window_budget = 10

    async def fake_compact(messages, model, cfg, logger, usage=None):
        if usage is not None:
            usage.update({"prompt_tokens": 800, "completion_tokens": 120})
        return [messages[0], {"role": "system", "content": "BRIEFING"}]

    mocker.patch.object(agent_mod, "_compact_messages", fake_compact)
    mocker.patch.object(agent_mod, "chat", side_effect=usage_reply("done", 100, 10))
    await agent.run("a task long enough to exceed the tiny budget", client=client)
    assert agent.tokens == {"prompt": 900, "completion": 130}


async def test_manual_compaction_is_counted(agent, mocker):
    async def fake_compact(messages, model, cfg, logger, usage=None):
        if usage is not None:
            usage.update({"prompt_tokens": 500, "completion_tokens": 70})
        return messages[:1]

    sid = agent.store.create_session("/p", "m", "t")
    for i in range(30):
        agent.store.append_message(sid, i, {"role": "user", "content": f"m{i}"})
    mocker.patch.object(agent_mod, "_compact_messages", fake_compact)
    await agent.compact_history(sid)
    assert agent.tokens == {"prompt": 500, "completion": 70}


async def test_a_server_that_reports_no_usage_counts_nothing(agent, client, mocker):
    """Not every OpenAI-compatible server sends a usage block."""
    replies(mocker, text_reply("done"))
    await agent.run("t", client=client)
    assert agent.tokens == {"prompt": 0, "completion": 0}


async def test_each_agent_counts_only_its_own(cfg, client, mocker):
    """Subagents have their own tally, which is what the pane's spinner
    shows — main's number must not absorb theirs."""
    mocker.patch.object(agent_mod.ui, "step_display")
    mocker.patch.object(agent_mod.ui, "elapsed_note")
    mocker.patch.object(agent_mod.ui, "banner")
    first, second = CodingAgent(cfg), CodingAgent(cfg)
    mocker.patch.object(agent_mod, "chat", side_effect=usage_reply("done", 100, 10))
    await first.run("one", client=client)
    assert first.tokens == {"prompt": 100, "completion": 10}
    assert second.tokens == {"prompt": 0, "completion": 0}


# ---------------- several subagents at once ----------------

async def test_two_spawns_in_one_step_run_concurrently(agent, client, mocker):
    """This is how more than one subagent comes about: the model issues
    several spawn_agent calls in a single turn. spawn_agent is a safe tool, so
    the step executor runs them together rather than one after another —
    otherwise two 3-second subagents would cost six seconds."""
    started = []

    async def slow_spawn(name, args):
        started.append(name)
        await asyncio.sleep(0.3)
        return f"{args.get('name')} reported"

    client.call_tool = mocker.AsyncMock(side_effect=slow_spawn)
    replies(mocker,
            tool_reply(("spawn_agent", '{"task": "a", "name": "one"}'),
                        ("spawn_agent", '{"task": "b", "name": "two"}')),
            text_reply("both reported"))

    start = time.monotonic()
    await agent.run("split it", client=client)
    elapsed = time.monotonic() - start

    assert len(started) == 2
    assert elapsed < 0.55, f"ran one after another ({elapsed:.2f}s for two 0.3s calls)"


async def test_spawn_needs_no_approval_but_what_it_does_still_might(cfg, mocker):
    """Spawning changes nothing by itself, so it isn't gated; the subagent's
    own writes are approved on their own terms."""
    from omni.agent import _approve
    approve = mocker.patch.object(agent_mod.ui, "request_approval", mocker.AsyncMock())
    assert await _approve("spawn_agent", {}, cfg, mocker.AsyncMock()) is True
    approve.assert_not_called()
    await _approve("write_file", {}, cfg, mocker.AsyncMock())
    approve.assert_awaited_once()


# ---------------- lines typed while a turn is running ----------------
#
# The input row stays live during a turn (see test_tui.py), and what you type
# there is handed to the agent running it — agent.inject. It becomes an
# ordinary user message in the conversation the model is reasoning from,
# folded in at the next step boundary, so the turn acts on it rather than
# queueing it behind itself.


def roles_of(conversation):
    return [m["role"] for m in conversation]


def texts_of(conversation, role):
    return [str(m.get("content")) for m in conversation if m["role"] == role]


def inject_from_the_tool_call(agent, client, *texts, result="file contents"):
    """Make the next tool call be the moment the person types."""
    async def run(*a, **k):
        for text in texts:
            agent.inject(text)
        return result
    client.call_tool.side_effect = run


def inject_after_the_first_reply(agent, mocker, *texts):
    """Make the model's first reply be the moment the person types — the
    reply is already written, so the turn is on the edge of ending."""
    real = agent_mod.chat
    seen = {"n": 0}

    async def chat_then_inject(*args, **kwargs):
        reply = await real(*args, **kwargs)
        seen["n"] += 1
        if seen["n"] == 1:
            for text in texts:
                agent.inject(text)
        return reply

    mocker.patch.object(agent_mod, "chat", chat_then_inject)


async def test_an_injected_line_reaches_the_model_in_the_same_turn(agent, client, mocker):
    """The whole point: acted on now, not in a turn of its own afterwards."""
    m = replies(mocker, tool_reply(("read_file", '{"path": "x.py"}')), text_reply("done"))
    agent.cfg.auto_approve = True
    inject_from_the_tool_call(agent, client, "also check the tests")

    await agent.run("read x.py", client=client)

    assert "also check the tests" in texts_of(m.sent[-1], "user")
    assert len(m.sent) == 2                # one turn, two model calls


async def test_the_injected_line_lands_after_the_tool_result(agent, client, mocker):
    """A user message between an assistant's tool_calls and the tool results
    that answer them is rejected outright by a strict server, so the fold-in
    happens on a step boundary and nowhere else."""
    m = replies(mocker, tool_reply(("read_file", '{"path": "x.py"}')), text_reply("done"))
    agent.cfg.auto_approve = True
    inject_from_the_tool_call(agent, client, "and the tests")

    await agent.run("read x.py", client=client)

    assert roles_of(m.sent[-1]) == ["system", "user", "assistant", "tool", "user"]


async def test_a_line_typed_during_the_final_reply_keeps_the_turn_alive(agent, client, mocker):
    """That reply carried no tool calls, so the turn was about to end.
    Ending it would answer a question that had just been added to."""
    m = replies(mocker, text_reply("I will use SQLite."), text_reply("Switched to Postgres."))
    inject_after_the_first_reply(agent, mocker, "actually use Postgres")

    result = await agent.run("set up the store", client=client)

    assert result == "Switched to Postgres."
    assert texts_of(m.sent[-1], "user") == ["set up the store", "actually use Postgres"]


async def test_the_interim_reply_is_not_swallowed_when_the_turn_continues(agent, client, mocker):
    """It was going to be the answer, so nothing had shown it yet. It turns
    out to be interim, so it has to be shown as interim."""
    replies(mocker, text_reply("I will use SQLite."), text_reply("Switched."))
    shown = mocker.patch("omni.ui.assistant_message")
    inject_after_the_first_reply(agent, mocker, "actually use Postgres")

    await agent.run("set up the store", client=client)
    assert any("SQLite" in call.args[0] for call in shown.call_args_list)


async def test_an_injected_line_is_persisted_so_a_resume_replays_it(agent, client, mocker):
    """It is part of the conversation now, not a UI flourish."""
    replies(mocker, tool_reply(("read_file", '{"path": "x.py"}')), text_reply("done"))
    agent.cfg.auto_approve = True
    inject_from_the_tool_call(agent, client, "and the tests")

    await agent.run("read x.py", client=client)

    stored = agent.store.load_messages(agent.session_id)
    assert texts_of(stored, "user") == ["read x.py", "and the tests"]


async def test_several_injected_lines_are_folded_in_the_order_typed(agent, client, mocker):
    m = replies(mocker, tool_reply(("read_file", '{"path": "x"}')), text_reply("done"))
    agent.cfg.auto_approve = True
    inject_from_the_tool_call(agent, client, "first", "second", "third")

    await agent.run("go", client=client)
    assert texts_of(m.sent[-1], "user") == ["go", "first", "second", "third"]


async def test_an_injected_image_travels_with_its_line(agent, client, mocker):
    m = replies(mocker, tool_reply(("read_file", '{"path": "x"}')), text_reply("done"))
    agent.cfg.auto_approve = True

    async def inject_with_image(*a, **k):
        agent.inject("what is wrong here", [{"mime": "image/png", "data": b"shot"}])
        return "contents"

    client.call_tool.side_effect = inject_with_image
    await agent.run("go", client=client)

    injected = [msg for msg in m.sent[-1] if msg["role"] == "user"][-1]
    assert isinstance(injected["content"], list)
    assert any(part.get("type") == "image_url" for part in injected["content"])


@pytest.mark.parametrize("nothing", ["", "   ", "\n", None])
def test_an_empty_line_is_nothing_to_say(agent, nothing):
    """Not an error — there is simply nothing to hand over."""
    assert agent.inject(nothing) is False
    assert agent.inbox_depth == 0


def test_inject_reports_that_it_took_the_line(agent):
    """cli believes only an exact True: a stub agent answers every call with
    something truthy, and trusting one would lose the line for real."""
    assert agent.inject("something") is True
    assert agent.inbox_depth == 1


def test_take_inbox_empties_it(agent):
    agent.inject("one")
    agent.inject("two")
    assert [text for text, _ in agent.take_inbox()] == ["one", "two"]
    assert agent.inbox_depth == 0
    assert agent.take_inbox() == []


async def test_a_line_that_arrived_too_late_is_left_for_the_caller(agent, client, mocker):
    """Injected after the last boundary, so the turn never saw it. It has to
    stay recoverable — cli moves it onto the pane's queue."""
    replies(mocker, text_reply("done"))
    result = await agent.run("go", client=client)
    agent.inject("too late")                 # the turn is already over
    assert result == "done"
    assert [text for text, _ in agent.take_inbox()] == ["too late"]


async def test_the_inbox_does_not_leak_into_the_next_turn(agent, client, mocker):
    """Whatever a turn absorbed is gone, so turn two must not re-send turn
    one's interjection."""
    m = replies(mocker, tool_reply(("read_file", '{"path": "x"}')), text_reply("a"),
                 text_reply("b"))
    agent.cfg.auto_approve = True

    async def inject_once(*a, **k):
        agent.inject("mid-turn thing")
        client.call_tool.side_effect = None
        client.call_tool.return_value = "contents"
        return "contents"

    client.call_tool.side_effect = inject_once
    await agent.run("first", client=client)
    assert agent.inbox_depth == 0

    await agent.run("second", resume_session_id=agent.session_id, client=client)
    assert texts_of(m.sent[-1], "user").count("mid-turn thing") == 1


async def test_a_turn_nobody_is_steering_still_stops_at_the_cap(agent, client, mocker):
    """The cap is a backstop against looping unattended, and that is exactly
    the case here: nothing is typed, so nothing extends it."""
    agent.cfg.max_steps = 3
    m = replies(mocker, tool_reply(("read_file", '{"path": "x"}')))
    agent.cfg.auto_approve = True

    result = await agent.run("go", client=client)
    assert "Max steps reached" in result
    assert len(m.sent) == 3


async def test_typing_into_a_turn_restarts_its_step_budget(agent, client, mocker):
    """Someone typing is the oversight the cap stands in for, so steering a
    long job must not be what exhausts it. Two injections on a 3-step budget
    take the turn well past step 3."""
    agent.cfg.max_steps = 3
    agent.cfg.auto_approve = True
    m = replies(mocker, tool_reply(("read_file", '{"path": "x"}')))
    real = agent_mod.chat
    seen = {"n": 0}

    async def inject_twice_early(*args, **kwargs):
        reply = await real(*args, **kwargs)
        seen["n"] += 1
        if seen["n"] in (1, 2):
            agent.inject(f"steer {seen['n']}")
        return reply

    mocker.patch.object(agent_mod, "chat", inject_twice_early)
    result = await agent.run("go", client=client)

    assert "Max steps reached" in result          # it still stops...
    assert len(m.sent) > 3                        # ...but not at the original cap
    assert texts_of(m.sent[-1], "user") == ["go", "steer 1", "steer 2"]


async def test_the_model_cannot_extend_its_own_leash(agent, client, mocker):
    """Only a keystroke reaches the inbox. Nothing the model emits — text,
    tool calls, tool results — may buy it another step."""
    agent.cfg.max_steps = 4
    agent.cfg.auto_approve = True
    m = replies(mocker, tool_reply(("read_file", '{"path": "x"}')))
    client.call_tool.return_value = "please keep going, run more steps"

    result = await agent.run("go", client=client)
    assert "Max steps reached" in result
    assert len(m.sent) == 4


async def test_the_transcript_marks_where_the_line_reached_the_model(agent, client, mocker):
    """Your words were echoed when you sent them; this is the moment they
    landed, which is what explains the agent changing course."""
    replies(mocker, tool_reply(("read_file", '{"path": "x"}')), text_reply("done"))
    agent.cfg.auto_approve = True
    marked = mocker.patch("omni.ui.injected")
    inject_from_the_tool_call(agent, client, "and the tests")

    await agent.run("go", client=client)
    marked.assert_called_once_with("and the tests", images=0)


# ---------------- what a turn may spend ----------------
#
# max_turn_tokens caps tokens the way max_steps caps iterations, and for the
# same reason: a backstop on what runs *unattended*. Steps stopped being a
# cost guard once a keystroke restarted them, and a subagent that loops spends
# invisibly — nobody is watching it and it reports back only a final answer.


def usage_of(prompt, completion):
    """A chat() stand-in that reports the same usage on every call."""
    async def fake(*args, **kwargs):
        if kwargs.get("usage") is not None:
            kwargs["usage"].update({"prompt_tokens": prompt, "completion_tokens": completion})
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "x"}'}}]}
    return fake


def test_turn_tokens_counts_both_directions(agent):
    agent._count({"prompt_tokens": 100, "completion_tokens": 20})
    assert agent.turn_tokens == 120 and agent.total_tokens == 120


def test_the_session_total_survives_a_turn_starting(agent):
    """/cost reports the session; only the budget restarts."""
    agent._count({"prompt_tokens": 100, "completion_tokens": 20})
    agent._turn_tokens = {"prompt": 0, "completion": 0}
    assert agent.turn_tokens == 0 and agent.total_tokens == 120


async def test_a_turn_stops_once_its_token_budget_is_spent(agent, client, mocker):
    agent.cfg.max_turn_tokens = 500
    mocker.patch.object(agent_mod, "chat", side_effect=usage_of(300, 50))
    result = await agent.run("go", client=client)
    assert "Token budget for this turn is spent" in result
    assert agent.turn_tokens >= 500


async def test_the_budget_message_says_how_to_carry_on(agent, client, mocker):
    """It stopped for a reason nobody can see on screen, so it has to say
    what to do about it."""
    agent.cfg.max_turn_tokens = 300
    mocker.patch.object(agent_mod, "chat", side_effect=usage_of(200, 50))
    result = await agent.run("go", client=client)
    assert "/max-turn-tokens" in result and "carry on" in result


async def test_a_zero_budget_is_no_budget(agent, client, mocker):
    """The default: a local model costs nothing per token, and a limit
    nobody asked for is a turn that stops for no visible reason."""
    agent.cfg.max_turn_tokens = 0
    agent.cfg.max_steps = 3
    mocker.patch.object(agent_mod, "chat", side_effect=usage_of(10_000, 10_000))
    result = await agent.run("go", client=client)
    assert "Token budget" not in result


async def test_the_session_status_records_why_it_stopped(agent, client, mocker):
    agent.cfg.max_turn_tokens = 200
    mocker.patch.object(agent_mod, "chat", side_effect=usage_of(150, 100))
    await agent.run("go", client=client)
    row = agent.store.list_sessions()[0]
    assert row["status"] == "max_tokens"


async def test_typing_into_a_turn_restarts_its_token_budget(agent, client, mocker):
    """Same rule as the step cap, same reason: someone typing is the
    oversight the cap stands in for."""
    agent.cfg.max_turn_tokens = 400
    agent.cfg.max_steps = 12
    spend = usage_of(150, 50)
    seen = {"n": 0}

    async def chat_and_interject(*args, **kwargs):
        reply = await spend(*args, **kwargs)
        seen["n"] += 1
        if seen["n"] == 2:
            agent.inject("keep going, but in pkg/")
        return reply

    mocker.patch.object(agent_mod, "chat", side_effect=chat_and_interject)
    await agent.run("go", client=client)
    # Without the restart the turn would have stopped at 400; the
    # interjection bought it another full budget.
    assert agent.turn_tokens > 400


async def test_a_budget_already_spent_stops_before_calling_the_model_again(agent, client, mocker):
    agent.cfg.max_turn_tokens = 100
    m = mocker.patch.object(agent_mod, "chat", side_effect=usage_of(200, 0))
    await agent.run("go", client=client)
    assert m.call_count == 1          # spent on the first call, stopped before a second


# ---------------- typing while it reads ----------------


async def test_typing_abandons_read_only_calls_in_flight(agent, client, mocker):
    """Reads are the one thing worth dropping half-done: nothing has changed,
    the answer is about to be irrelevant, and waiting for eight file reads
    before the model hears "wrong directory" is the delay this is for."""
    started = asyncio.Event()

    async def never_finishes(name, args):
        started.set()
        await asyncio.sleep(30)
        return "too late"

    client.call_tool.side_effect = never_finishes
    replies(mocker, tool_reply(("read_file", '{"path": "x"}')), text_reply("stopped"))

    async def drive():
        turn = asyncio.ensure_future(agent.run("read it", client=client))
        await started.wait()
        await asyncio.sleep(0)
        agent.inject("wrong directory")
        return await turn

    result = await asyncio.wait_for(drive(), timeout=5)
    assert result == "stopped"


async def test_an_abandoned_call_says_it_was_superseded_not_interrupted(agent, client, mocker):
    """Why it stopped matters to the model: "the user said something" is
    followed by that something and reads as a change of direction, where
    "Ctrl+C" reads as a refusal."""
    started = asyncio.Event()

    async def never_finishes(name, args):
        started.set()
        await asyncio.sleep(30)

    client.call_tool.side_effect = never_finishes
    m = replies(mocker, tool_reply(("read_file", '{"path": "x"}')), text_reply("ok"))

    async def drive():
        turn = asyncio.ensure_future(agent.run("read it", client=client))
        await started.wait()
        await asyncio.sleep(0)
        agent.inject("wrong directory")
        return await turn

    await asyncio.wait_for(drive(), timeout=5)
    tool_messages = [m_ for m_ in m.sent[-1] if m_["role"] == "tool"]
    assert "superseded" in tool_messages[0]["content"]
    assert "Ctrl+C" not in tool_messages[0]["content"]


async def test_a_write_in_flight_is_never_torn_in_half(agent, client, mocker):
    """Abandoning a write is how you get a truncated file, so only the
    read-only branch is interruptible."""
    agent.cfg.auto_approve = True
    running = asyncio.Event()

    async def slow_write(name, args):
        running.set()
        await asyncio.sleep(0.05)
        return "written in full"

    client.call_tool.side_effect = slow_write
    m = replies(mocker, tool_reply(("write_file", '{"path": "x"}')), text_reply("done"))

    async def drive():
        turn = asyncio.ensure_future(agent.run("write it", client=client))
        await running.wait()
        agent.inject("actually stop")
        return await turn

    await asyncio.wait_for(drive(), timeout=5)
    tool_messages = [m_ for m_ in m.sent[-1] if m_["role"] == "tool"]
    assert tool_messages[0]["content"] == "written in full"


async def test_nothing_is_interruptible_between_steps(agent, client, mocker):
    """Outside a read, there is no in-flight call to abandon — injecting is
    just the ordinary fold-in."""
    replies(mocker, text_reply("done"))
    assert agent._tool_cancel is None
    agent.inject("hello")
    assert agent._tool_cancel is None


async def test_the_canceller_is_cleared_once_the_reads_finish(agent, client, mocker):
    """Left set, a later interjection would cancel tasks that are long gone."""
    replies(mocker, tool_reply(("read_file", '{"path": "x"}')), text_reply("done"))
    await agent.run("go", client=client)
    assert agent._tool_cancel is None
