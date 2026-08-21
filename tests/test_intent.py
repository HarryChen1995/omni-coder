"""intent.py — JSON coercion, context rendering, and the retry/fallback loop.

`chat` is mocked so the retry and give-up paths are exercised deterministically;
asyncio.sleep is stubbed so backoff doesn't slow the suite.
"""

import json

import pytest

from omni import intent as intent_mod
from omni.intent import Intent, _coerce, _strip_code_fence, extract_intent
from omni.llm_client import LLMError


@pytest.fixture(autouse=True)
def no_backoff_sleep(mocker):
    """extract_intent sleeps between retries — skip the real waiting."""
    mocker.patch.object(intent_mod.asyncio, "sleep", mocker.AsyncMock())


def chat_returning(mocker, *contents):
    """Patch chat() to yield each content in turn; an Exception entry is
    raised instead of returned, to drive the retry branches."""
    async def fake(*args, **kwargs):
        item = contents[min(fake.calls, len(contents) - 1)]
        fake.calls += 1
        if isinstance(item, BaseException):
            raise item
        return {"content": item} if not isinstance(item, dict) else item

    fake.calls = 0
    return mocker.patch.object(intent_mod, "chat", side_effect=fake)


# ---------------- _strip_code_fence ----------------

@pytest.mark.parametrize("raw", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '```\n{"a": 1}\n```',
    '   ```json\n{"a": 1}\n```   ',
])
def test_strip_code_fence(raw):
    assert json.loads(_strip_code_fence(raw)) == {"a": 1}


# ---------------- _coerce ----------------

def test_coerce_full_valid_payload():
    i = _coerce({"task_type": "bugfix", "summary": "fix add()",
                 "target_files": ["a.py"], "constraints": ["no deps"], "risk_level": "low"})
    assert (i.task_type, i.risk_level, i.confident) == ("bugfix", "low", True)
    assert i.target_files == ["a.py"] and i.constraints == ["no deps"]


@pytest.mark.parametrize("task_type", ["bugfix", "feature", "refactor", "test", "docs", "explore", "other"])
def test_coerce_accepts_each_valid_task_type(task_type):
    assert _coerce({"task_type": task_type}).task_type == task_type


@pytest.mark.parametrize("bad", ["nonsense", "", None, 42, [], {}, ["bugfix"]])
def test_coerce_unknown_task_type_falls_back_to_other(bad):
    """Includes unhashable list/dict values: `x in <set>` would raise
    TypeError on those, so they must be type-checked before the lookup."""
    assert _coerce({"task_type": bad}).task_type == "other"


@pytest.mark.parametrize("bad", ["extreme", None, 7, "", [], {}])
def test_coerce_unknown_risk_falls_back_to_medium(bad):
    """Unknown risk defaults to medium — never silently to 'low', which
    would skip the forced-approval path."""
    assert _coerce({"risk_level": bad}).risk_level == "medium"


def test_coerce_wraps_scalar_lists():
    i = _coerce({"target_files": "only.py", "constraints": "just one"})
    assert i.target_files == ["only.py"] and i.constraints == ["just one"]


def test_coerce_stringifies_list_entries():
    i = _coerce({"target_files": [1, None, 2.5], "constraints": [True]})
    assert i.target_files == ["1", "None", "2.5"] and i.constraints == ["True"]


def test_coerce_caps_list_length_and_summary_length():
    i = _coerce({"target_files": [f"f{n}.py" for n in range(50)],
                 "constraints": [f"c{n}" for n in range(50)],
                 "summary": "x" * 900})
    assert len(i.target_files) == 20 and len(i.constraints) == 20
    assert len(i.summary) == 500


def test_coerce_keeps_usable_fields_when_type_is_malformed():
    """A bad task_type must not throw away the rest of the payload."""
    i = _coerce({"task_type": [], "summary": "still useful", "target_files": ["a.py"]})
    assert i.task_type == "other" and i.summary == "still useful"
    assert i.target_files == ["a.py"] and i.confident is True


def test_coerce_missing_and_null_fields():
    i = _coerce({})
    assert i.task_type == "other" and i.risk_level == "medium" and i.summary == ""
    assert i.target_files == [] and i.constraints == []
    assert _coerce({"target_files": None, "constraints": None}).target_files == []


def test_coerce_keeps_raw_payload():
    payload = {"task_type": "docs", "extra": "kept"}
    assert _coerce(payload).raw == payload


# ---------------- as_context_block ----------------

def test_context_block_contains_all_fields():
    block = Intent(task_type="feature", summary="add x", target_files=["a.py"],
                   constraints=["fast"], risk_level="high").as_context_block()
    assert "task_type: feature" in block and "risk_level: high" in block
    assert "summary: add x" in block and "a.py" in block and "fast" in block
    assert "low confidence" not in block


def test_context_block_tags_existing_vs_new_files():
    block = Intent(target_files=["old.py", "new.py"]).as_context_block(
        {"old.py": True, "new.py": False})
    assert "old.py (exists)" in block and "new.py (new)" in block


def test_context_block_leaves_untagged_files_bare():
    """A file missing from `existing_files` is listed without an
    exists/new tag rather than guessing one."""
    block = Intent(target_files=["a.py", "b.py"]).as_context_block({"a.py": True})
    assert "target_files: a.py (exists), b.py" in block


def test_context_block_placeholders_when_empty():
    block = Intent().as_context_block()
    assert "none specified" in block and "none stated" in block


def test_context_block_flags_low_confidence():
    assert "low confidence" in Intent(confident=False).as_context_block()


# ---------------- extract_intent ----------------

async def test_extract_intent_parses_first_try(mocker):
    m = chat_returning(mocker, '{"task_type": "bugfix", "summary": "s", "risk_level": "low"}')
    i = await extract_intent("fix it", "model")
    assert i.task_type == "bugfix" and i.confident is True
    assert m.call_count == 1


async def test_extract_intent_requests_json_mode_with_the_schema_prompt(mocker):
    m = chat_returning(mocker, '{"task_type": "docs"}')
    await extract_intent("write docs", "my-model", base_url="http://h", api_key="k", timeout=42.0)
    kwargs = m.call_args.kwargs
    assert kwargs["format"] == "json" and kwargs["model"] == "my-model"
    assert kwargs["base_url"] == "http://h" and kwargs["api_key"] == "k"
    assert kwargs["timeout"] == 42.0
    assert kwargs["messages"][0]["content"] == intent_mod.INTENT_SCHEMA_PROMPT
    assert kwargs["messages"][1] == {"role": "user", "content": "write docs"}


async def test_extract_intent_strips_code_fences(mocker):
    chat_returning(mocker, '```json\n{"task_type": "test"}\n```')
    assert (await extract_intent("t", "m")).task_type == "test"


async def test_extract_intent_retries_after_bad_json_then_succeeds(mocker):
    m = chat_returning(mocker, "not json at all", '{"task_type": "feature"}')
    i = await extract_intent("t", "m")
    assert i.task_type == "feature" and i.confident is True
    assert m.call_count == 2


async def test_extract_intent_retries_after_llm_error(mocker):
    m = chat_returning(mocker, LLMError("503"), '{"task_type": "refactor"}')
    assert (await extract_intent("t", "m")).task_type == "refactor"
    assert m.call_count == 2


async def test_extract_intent_retries_after_missing_content_key(mocker):
    m = chat_returning(mocker, {"no_content": 1}, '{"task_type": "docs"}')
    assert (await extract_intent("t", "m")).task_type == "docs"
    assert m.call_count == 2


async def test_extract_intent_retries_after_unexpected_error(mocker):
    m = chat_returning(mocker, RuntimeError("boom"), '{"task_type": "explore"}')
    assert (await extract_intent("t", "m")).task_type == "explore"
    assert m.call_count == 2


async def test_extract_intent_gives_up_with_low_confidence_fallback(mocker):
    """After max_retries it must degrade, not raise — the agent still runs
    on the raw task text."""
    m = chat_returning(mocker, "garbage")
    i = await extract_intent("the original task text", "m", max_retries=3)
    assert m.call_count == 3
    assert i.confident is False and i.task_type == "other"
    assert i.summary == "the original task text"


async def test_extract_intent_fallback_truncates_long_task(mocker):
    chat_returning(mocker, "garbage")
    i = await extract_intent("y" * 500, "m", max_retries=1)
    assert len(i.summary) == 200


async def test_extract_intent_honors_max_retries(mocker):
    m = chat_returning(mocker, LLMError("down"))
    await extract_intent("t", "m", max_retries=1)
    assert m.call_count == 1


async def test_extract_intent_logs_each_outcome(mocker):
    logger = mocker.Mock()
    chat_returning(mocker, "bad", '{"task_type": "bugfix"}')
    await extract_intent("t", "m", logger=logger)
    logged = " ".join(str(c) for c in logger.info.call_args_list)
    assert "INTENT parse failed" in logged and "INTENT parsed" in logged


async def test_extract_intent_logs_giving_up(mocker):
    logger = mocker.Mock()
    chat_returning(mocker, "bad")
    await extract_intent("t", "m", max_retries=2, logger=logger)
    assert "gave up" in " ".join(str(c) for c in logger.info.call_args_list)
