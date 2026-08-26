"""agent.py's pure helpers: JSON recovery, history trimming, elapsed
formatting, memory loading, approval policy, and LLM-based compaction."""

import pytest

from omni import agent as agent_mod
from omni.agent import (
    _approve, _compact_messages, _ensure_tool_call_ids, _find_json_objects,
    _quiet_library_logs,
    _format_elapsed, _load_project_memory, _protected_head_len,
    _recover_text_tool_calls, _render_for_summary, _setup_logger, _trim_history,
)
from omni.llm_client import LLMError


# ---------------- _load_project_memory ----------------

def test_load_project_memory_reads_and_strips(project_root):
    (project_root / "agent_memory.md").write_text("\n- uses pytest\n\n")
    assert _load_project_memory(str(project_root), "agent_memory.md") == "- uses pytest"


def test_load_project_memory_missing_file_is_empty(project_root):
    assert _load_project_memory(str(project_root), "nope.md") == ""


# ---------------- _setup_logger ----------------

def test_setup_logger_writes_to_the_given_path(tmp_path):
    path = tmp_path / "run.log"
    logger = _setup_logger(str(path))
    logger.info("a message")
    for h in logger.handlers:
        h.flush()
    assert "a message" in path.read_text()


def test_setup_logger_replaces_handlers_on_reconfigure(tmp_path):
    """Called once per CodingAgent; handlers must not accumulate and write
    duplicate lines."""
    _setup_logger(str(tmp_path / "a.log"))
    logger = _setup_logger(str(tmp_path / "b.log"))
    assert len(logger.handlers) == 1


# ---------------- _find_json_objects ----------------

def test_find_json_objects_single_and_multiple():
    assert _find_json_objects('{"a": 1}') == ['{"a": 1}']
    assert _find_json_objects('x {"a":1} y {"b":2} z') == ['{"a":1}', '{"b":2}']


def test_find_json_objects_handles_nesting():
    assert _find_json_objects('{"a": {"b": {"c": 1}}}') == ['{"a": {"b": {"c": 1}}}']


def test_find_json_objects_ignores_braces_inside_strings():
    """Code the model is writing often contains braces — they must not throw
    off the balance count."""
    src = '{"content": "def f() { return {1:2}; }"}'
    assert _find_json_objects(src) == [src]


def test_find_json_objects_handles_escaped_quotes():
    src = r'{"content": "say \"hi\" {"}'
    assert _find_json_objects(src) == [src]


def test_find_json_objects_none_and_unbalanced():
    assert _find_json_objects("no braces here") == []
    assert _find_json_objects("") == []
    assert _find_json_objects('{"unclosed": 1') == []


# ---------------- _recover_text_tool_calls ----------------

NAMES = {"read_file", "write_file"}


def test_recover_tool_call_from_plain_text():
    calls = _recover_text_tool_calls('{"name": "read_file", "arguments": {"path": "x.py"}}', NAMES)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert calls[0]["function"]["arguments"] == '{"path": "x.py"}'
    assert calls[0]["type"] == "function" and calls[0]["id"].startswith("fallback_")


def test_recover_multiple_calls_get_distinct_ids():
    text = ('{"name": "read_file", "arguments": {}} and '
            '{"name": "write_file", "arguments": {"path": "y"}}')
    calls = _recover_text_tool_calls(text, NAMES)
    assert len({c["id"] for c in calls}) == 2


def test_recover_ignores_unknown_tool_names():
    assert _recover_text_tool_calls('{"name": "rm_rf", "arguments": {}}', NAMES) == []


def test_recover_ignores_non_dict_arguments():
    assert _recover_text_tool_calls('{"name": "read_file", "arguments": "x"}', NAMES) == []


def test_recover_requires_both_keys():
    assert _recover_text_tool_calls('{"name": "read_file"}', NAMES) == []
    assert _recover_text_tool_calls('{"arguments": {}}', NAMES) == []


@pytest.mark.parametrize("content", ["", None, "just prose, no json"])
def test_recover_short_circuits_on_no_json(content):
    assert _recover_text_tool_calls(content, NAMES) == []


def test_recover_finds_call_embedded_in_prose():
    text = 'Sure! I will read it:\n{"name": "read_file", "arguments": {"path": "a"}}\nDone.'
    assert len(_recover_text_tool_calls(text, NAMES)) == 1


# ---------------- _format_elapsed ----------------

@pytest.mark.parametrize("seconds,expected", [
    (0.0, "0.0s"), (3.24, "3.2s"), (59.9, "59.9s"),
    (60, "1m 00s"), (61.4, "1m 01s"), (599, "9m 59s"),
    (3600, "1h 00m 00s"), (3725, "1h 02m 05s"),
])
def test_format_elapsed(seconds, expected):
    assert _format_elapsed(seconds) == expected


def test_agent_and_ui_elapsed_formats_agree():
    """agent._format_elapsed is a deliberate duplicate of ui's (agent must
    work without rich) — they must not drift."""
    from omni.ui import _format_elapsed as ui_fmt
    for s in (0.5, 12.34, 60, 90, 3600, 7325):
        assert _format_elapsed(s) == ui_fmt(s)


# ---------------- _trim_history ----------------

def msgs(n, size=10):
    return ([{"role": "system", "content": "s" * size}, {"role": "user", "content": "u" * size}]
            + [{"role": "assistant", "content": f"m{i}" * size} for i in range(n)])


def test_trim_history_under_budget_is_unchanged():
    m = msgs(2)
    assert _trim_history(m, 10_000) is m


def test_trim_history_drops_oldest_but_keeps_system_and_task():
    m = msgs(20, size=100)
    out = _trim_history(m, 500)
    assert out[0]["role"] == "system" and out[1]["role"] == "user"
    assert len(out) < len(m)
    assert out[-1] == m[-1]        # newest kept


def test_trim_history_tolerates_missing_content():
    m = [{"role": "system"}, {"role": "user"}, {"role": "assistant", "content": None}]
    assert _trim_history(m, 0) == m[:2]


# ---------------- _render_for_summary ----------------

def test_render_for_summary_plain_message():
    assert _render_for_summary({"role": "user", "content": "hello"}) == "user: hello"


def test_render_for_summary_renders_tool_calls():
    out = _render_for_summary({"role": "assistant", "tool_calls": [
        {"function": {"name": "read_file", "arguments": '{"path": "x"}'}}]})
    assert out == 'assistant: called read_file({"path": "x"})'


def test_render_for_summary_truncates_long_content():
    out = _render_for_summary({"role": "tool", "content": "z" * 2000}, max_len=50)
    assert len(out) < 100 and out.endswith("…")


def test_render_for_summary_handles_none_content():
    assert _render_for_summary({"role": "assistant", "content": None}) == "assistant: "


# ---------------- _approve policy ----------------

async def test_approve_allows_safe_tools_without_prompting(cfg, mocker):
    ui_mock = mocker.patch.object(agent_mod.ui, "request_approval", mocker.AsyncMock())
    assert await _approve("read_file", {}, cfg, None) is True
    ui_mock.assert_not_awaited()


async def test_approve_auto_approves_when_configured(cfg, mocker):
    cfg.auto_approve = True
    ui_mock = mocker.patch.object(agent_mod.ui, "request_approval", mocker.AsyncMock())
    assert await _approve("write_file", {}, cfg, None) is True
    ui_mock.assert_not_awaited()


async def test_force_approval_overrides_auto_approve(cfg, mocker):
    """A high-risk intent must still prompt even under --auto-approve."""
    cfg.auto_approve = True
    ui_mock = mocker.patch.object(agent_mod.ui, "request_approval", mocker.AsyncMock(return_value=False))
    assert await _approve("run_shell", {}, cfg, None, force_approval=True) is False
    ui_mock.assert_awaited_once()


async def test_approve_defers_to_the_ui_for_unsafe_tools(cfg, mocker):
    ui_mock = mocker.patch.object(agent_mod.ui, "request_approval", mocker.AsyncMock(return_value=True))
    assert await _approve("edit_file", {"path": "x"}, cfg, None) is True
    assert ui_mock.await_args.args[0] == "edit_file"


@pytest.mark.parametrize("typed,expected", [("y", True), ("Y", True), ("n", False), ("", False)])
async def test_approve_falls_back_to_stdin_without_rich(cfg, mocker, typed, expected):
    mocker.patch.object(agent_mod, "_HAS_UI", False)
    mocker.patch("builtins.input", return_value=typed)
    assert await _approve("write_file", {"path": "x"}, cfg, None) is expected


# ---------------- _compact_messages ----------------

def long_history(n=30):
    return ([{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
            + [{"role": "assistant", "content": f"step {i}"} for i in range(n)])


async def test_compact_short_history_is_untouched(cfg, mocker):
    m = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    assert await _compact_messages(m, "model", cfg, mocker.Mock()) is m


async def test_compact_replaces_the_middle_with_a_summary(cfg, mocker):
    cfg.compact_keep_last = 5
    mocker.patch.object(agent_mod, "chat", mocker.AsyncMock(return_value={"content": "THE SUMMARY"}))
    history = long_history(30)
    out = await _compact_messages(history, "model", cfg, mocker.Mock())

    assert out[0] == history[0] and out[1] == history[1]     # system + task kept verbatim
    assert out[2]["role"] == "system" and "THE SUMMARY" in out[2]["content"]
    assert "Compacted history" in out[2]["content"]
    assert out[-5:] == history[-5:]                          # recent tail kept verbatim
    assert len(out) == 2 + 1 + 5


async def test_compact_sends_a_rendered_transcript(cfg, mocker):
    cfg.compact_keep_last = 2
    chat_mock = mocker.patch.object(agent_mod, "chat", mocker.AsyncMock(return_value={"content": "s"}))
    await _compact_messages(long_history(10), "model", cfg, mocker.Mock())
    sent = chat_mock.await_args.kwargs["messages"]
    assert sent[0]["content"] == agent_mod._COMPACT_PROMPT
    assert "assistant: step 0" in sent[1]["content"]


async def test_compact_uses_the_configured_timeout_and_host(cfg, mocker):
    cfg.compact_keep_last, cfg.llm_timeout_s = 2, 77.0
    cfg.llm_host, cfg.llm_api_key = "http://h", "k"
    chat_mock = mocker.patch.object(agent_mod, "chat", mocker.AsyncMock(return_value={"content": "s"}))
    await _compact_messages(long_history(10), "m", cfg, mocker.Mock())
    kwargs = chat_mock.await_args.kwargs
    assert kwargs["timeout"] == 77.0 and kwargs["base_url"] == "http://h" and kwargs["api_key"] == "k"


async def test_compact_falls_back_to_trim_when_the_call_fails(cfg, mocker):
    cfg.compact_keep_last = 5
    cfg.context_char_budget = 50
    mocker.patch.object(agent_mod, "chat", mocker.AsyncMock(side_effect=LLMError("down")))
    logger = mocker.Mock()
    history = long_history(40)
    out = await _compact_messages(history, "model", cfg, logger)

    assert len(out) < len(history)
    assert not any("Compacted history" in str(m.get("content")) for m in out)  # trimmed, not summarized
    assert "falling back" in str(logger.info.call_args)


async def test_compact_falls_back_when_summary_is_empty(cfg, mocker):
    cfg.compact_keep_last, cfg.context_char_budget = 5, 50
    mocker.patch.object(agent_mod, "chat", mocker.AsyncMock(return_value={"content": "   "}))
    out = await _compact_messages(long_history(40), "model", cfg, mocker.Mock())
    assert not any("Compacted history" in str(m.get("content")) for m in out)


async def test_compact_logs_and_reports_elapsed(cfg, mocker):
    cfg.compact_keep_last = 2
    mocker.patch.object(agent_mod, "chat", mocker.AsyncMock(return_value={"content": "sum"}))
    ui_mock = mocker.patch.object(agent_mod.ui, "compacted")
    logger = mocker.Mock()
    await _compact_messages(long_history(10), "model", cfg, logger)
    assert "compacted" in str(logger.info.call_args)
    ui_mock.assert_called_once()
    assert ui_mock.call_args.args[0] == 8   # messages folded into the summary


# ---------------- _recover_text_tool_calls: narration guard ----------------

def test_recover_ignores_a_call_merely_described_in_a_final_answer():
    """The recovery path executes what it finds, so a summary that quotes the
    call it already made must not be re-run as a fresh one."""
    text = (
        "All done. I fixed the off-by-one in the loop bound, added a regression "
        "test for the empty-input case, and re-ran the suite (42 passed). The "
        "only tool call that touched disk was "
        '{"name": "write_file", "arguments": {"path": "x.py", "content": "boom"}}'
        " — everything else was read-only. To verify, run `pytest -q tests/test_x.py`; "
        "if it fails, check that the fixture still seeds three rows."
    )
    assert _recover_text_tool_calls(text, NAMES) == []


def test_recover_still_fires_with_a_short_lead_in():
    text = 'Reading that now.\n{"name": "read_file", "arguments": {"path": "a"}}'
    assert len(_recover_text_tool_calls(text, NAMES)) == 1


def test_recover_still_fires_inside_a_json_fence():
    text = '```json\n{"name": "read_file", "arguments": {"path": "a"}}\n```'
    assert len(_recover_text_tool_calls(text, NAMES)) == 1


# ---------------- _ensure_tool_call_ids ----------------

def test_ensure_tool_call_ids_fills_only_missing_ones():
    calls = [{"id": "abc", "function": {}}, {"function": {}}, {"id": "", "function": {}}]
    _ensure_tool_call_ids(calls, step=3)
    assert calls[0]["id"] == "abc"                    # server-supplied id untouched
    assert calls[1]["id"] == "call_3_1"
    assert calls[2]["id"] == "call_3_2"
    assert len({c["id"] for c in calls}) == 3


# ---------------- _protected_head_len ----------------

def test_protected_head_covers_the_injected_intent_block():
    """The regression this exists for: with intent parsing on, the task sits
    at index 2, so a fixed head of 2 summarized away the task itself."""
    m = [{"role": "system"}, {"role": "system"}, {"role": "user"}, {"role": "assistant"}]
    assert _protected_head_len(m) == 3


def test_protected_head_without_an_intent_block():
    assert _protected_head_len([{"role": "system"}, {"role": "user"}, {"role": "tool"}]) == 2


def test_protected_head_with_no_system_prompt():
    assert _protected_head_len([{"role": "user"}, {"role": "assistant"}]) == 1


def test_protected_head_of_system_only_and_empty():
    assert _protected_head_len([{"role": "system"}]) == 1
    assert _protected_head_len([]) == 0


def test_trim_history_keeps_the_task_behind_an_intent_block():
    m = ([{"role": "system", "content": "s" * 50}, {"role": "system", "content": "intent" * 20},
          {"role": "user", "content": "THE TASK"}]
         + [{"role": "assistant", "content": "x" * 200} for _ in range(20)])
    out = _trim_history(m, 500)
    assert [x["content"] for x in out[:3]] == ["s" * 50, "intent" * 20, "THE TASK"]
    assert len(out) < len(m)


async def test_compact_keeps_the_task_behind_an_intent_block(cfg, mocker):
    cfg.compact_keep_last = 3
    mocker.patch.object(agent_mod, "chat", mocker.AsyncMock(return_value={"content": "SUM"}))
    history = ([{"role": "system", "content": "sys"}, {"role": "system", "content": "[Parsed intent]"},
                {"role": "user", "content": "THE TASK"}]
               + [{"role": "assistant", "content": f"step {i}"} for i in range(20)])
    out = await _compact_messages(history, "model", cfg, mocker.Mock())
    assert [m["content"] for m in out[:3]] == ["sys", "[Parsed intent]", "THE TASK"]
    assert "SUM" in out[3]["content"]
    assert out[-3:] == history[-3:]
    assert len(out) == 3 + 1 + 3


async def test_compact_with_keep_last_zero_summarizes_everything_after_the_head(cfg, mocker):
    """messages[-0:] is the whole list, so a zero here used to duplicate the
    entire history behind the summary instead of dropping it."""
    cfg.compact_keep_last = 0
    mocker.patch.object(agent_mod, "chat", mocker.AsyncMock(return_value={"content": "SUM"}))
    history = ([{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
               + [{"role": "assistant", "content": f"step {i}"} for i in range(10)])
    out = await _compact_messages(history, "model", cfg, mocker.Mock())
    assert len(out) == 3 and "SUM" in out[2]["content"]


# ---------------- _approve: interrupt at the prompt ----------------

@pytest.mark.parametrize("boom", [KeyboardInterrupt(), EOFError()])
async def test_approve_treats_an_interrupted_prompt_as_denial(cfg, mocker, boom):
    mocker.patch.object(agent_mod.ui, "request_approval", mocker.AsyncMock(side_effect=boom))
    assert await _approve("write_file", {}, cfg, mocker.AsyncMock()) is False


# ---------------- library logging ----------------

def test_library_logs_go_to_the_file_not_the_terminal(tmp_path):
    """Without a handler of their own, third-party records reach stderr via
    Python's last-resort handler — i.e. the middle of the UI. The mcp package
    logs there when a transport struggles to reap a subprocess."""
    import logging

    log = tmp_path / "run.log"
    _quiet_library_logs(str(log))
    mcp_logger = logging.getLogger("mcp.os.posix.utilities")
    mcp_logger.warning("Process group termination failed for PID 1")

    assert mcp_logger.getEffectiveLevel() <= logging.WARNING
    assert logging.getLogger("mcp").propagate is False
    for handler in logging.getLogger("mcp").handlers:
        handler.flush()
    assert "Process group termination failed" in log.read_text()


def test_quiet_library_logs_is_idempotent(tmp_path):
    import logging

    log = tmp_path / "run.log"
    _quiet_library_logs(str(log))
    _quiet_library_logs(str(log))
    assert len(logging.getLogger("mcp").handlers) == 1   # no handler pile-up
