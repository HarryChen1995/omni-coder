"""Turn a freeform user request into structured intent — task type, target
files, constraints, risk level — before the agent starts taking actions.

This runs as a single, tool-free model call in strict JSON mode. It is
deliberately separate from the main agent loop: if it fails or the model
guesses wrong, the agent still runs — it just does so with less upfront
context, and a human reviewing the log can see exactly what was inferred.
"""

import json
import re
import asyncio
from dataclasses import dataclass, field
from typing import List

from .llm_client import chat, LLMError

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(content: str) -> str:
    """Models sometimes wrap JSON-mode output in a ```json fence despite
    being asked for a bare object — strip it before parsing."""
    return _CODE_FENCE_RE.sub("", content.strip())

INTENT_SCHEMA_PROMPT = """You are a task-intent parser for a coding agent. \
Given a user's freeform request, output ONLY a JSON object (no markdown \
fences, no commentary, no explanation) with exactly these keys:

{
  "task_type": one of "bugfix" | "feature" | "refactor" | "test" | "docs" | "explore" | "other",
  "summary": a single sentence restating the task in your own words,
  "target_files": array of file paths the user mentioned or clearly implied — best guess, can be empty,
  "constraints": array of explicit requirements or limits the user stated (e.g. "don't touch the tests", "keep it under 100 lines") — can be empty,
  "risk_level": "low" | "medium" | "high" — use "high" if the task involves deleting data, deploying, running migrations, force-pushing, or other hard-to-reverse actions
}

Respond with the JSON object only."""

_VALID_TASK_TYPES = {"bugfix", "feature", "refactor", "test", "docs", "explore", "other"}
_VALID_RISK = {"low", "medium", "high"}


@dataclass
class Intent:
    task_type: str = "other"
    summary: str = ""
    target_files: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    risk_level: str = "medium"
    confident: bool = True   # False if parsing fell back to the default
    raw: dict = field(default_factory=dict)

    def as_context_block(self, existing_files: dict = None) -> str:
        """Render as a short block to inject into the agent's messages.
        `existing_files` optionally maps target file -> bool(exists) so the
        agent knows upfront which targets are new vs. edits."""
        files_line = "none specified"
        if self.target_files:
            parts = []
            for f in self.target_files:
                if existing_files is not None and f in existing_files:
                    tag = "exists" if existing_files[f] else "new"
                    parts.append(f"{f} ({tag})")
                else:
                    parts.append(f)
            files_line = ", ".join(parts)

        constraints_line = "; ".join(self.constraints) if self.constraints else "none stated"
        confidence_note = "" if self.confident else " [low confidence — intent parsing fell back to defaults]"

        return (
            f"[Parsed intent]{confidence_note}\n"
            f"task_type: {self.task_type}\n"
            f"risk_level: {self.risk_level}\n"
            f"summary: {self.summary}\n"
            f"target_files: {files_line}\n"
            f"constraints: {constraints_line}"
        )


def _coerce(data: dict) -> Intent:
    # isinstance(str) first: a model can return a list/dict here, and an
    # unhashable value raises TypeError on `in <set>` — which would burn
    # every retry and degrade to the low-confidence fallback for a payload
    # whose other fields were perfectly usable.
    raw_type, raw_risk = data.get("task_type"), data.get("risk_level")
    task_type = raw_type if isinstance(raw_type, str) and raw_type in _VALID_TASK_TYPES else "other"
    risk_level = raw_risk if isinstance(raw_risk, str) and raw_risk in _VALID_RISK else "medium"

    target_files = data.get("target_files") or []
    if not isinstance(target_files, list):
        target_files = [target_files]

    constraints = data.get("constraints") or []
    if not isinstance(constraints, list):
        constraints = [constraints]

    return Intent(
        task_type=task_type,
        summary=str(data.get("summary", ""))[:500],
        target_files=[str(f) for f in target_files][:20],
        constraints=[str(c) for c in constraints][:20],
        risk_level=risk_level,
        confident=True,
        raw=data,
    )


async def extract_intent(task: str, model: str, max_retries: int = 3, logger=None, base_url: str = None,
                          api_key: str = None, timeout: float = 300.0) -> Intent:
    """Ask the model to parse `task` into structured intent. On repeated
    failure, returns a low-confidence Intent(task_type='other') rather than
    raising — callers should treat `confident=False` as a signal to fall
    back to plain freeform behavior, not as ground truth to act on."""
    messages = [
        {"role": "system", "content": INTENT_SCHEMA_PROMPT},
        {"role": "user", "content": task},
    ]

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            message = await chat(model=model, messages=messages, format="json", base_url=base_url,
                                  api_key=api_key, timeout=timeout)
            content = message["content"]
            data = json.loads(_strip_code_fence(content))
            intent = _coerce(data)
            if logger:
                logger.info(f"INTENT parsed (attempt {attempt}): {data}")
            return intent
        except (json.JSONDecodeError, KeyError, LLMError) as e:
            last_err = e
            if logger:
                logger.info(f"INTENT parse failed (attempt {attempt}): {e}")
            await asyncio.sleep(min(2 ** attempt, 5))
        except Exception as e:
            last_err = e
            if logger:
                logger.info(f"INTENT unexpected error (attempt {attempt}): {e}")
            await asyncio.sleep(1)

    if logger:
        logger.info(f"INTENT parsing gave up after {max_retries} attempts ({last_err}); using fallback.")
    return Intent(summary=task[:200], confident=False, raw={})
