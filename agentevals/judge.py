"""Model-graded checks, with the controls that make them trustworthy.

* :class:`RubricJudge` grades one answer against a rubric. The prompt asks for a
  JSON verdict; unparseable replies count as a fail (never a silent pass).
* :func:`pairwise` compares two answers **twice with the order swapped**. LLM
  judges have a measurable position bias; a preference that flips when the order
  flips is reported as a tie instead of a win.
* :func:`judge_agreement` measures how often a judge agrees with human labels
  (Cohen's kappa), which should be checked before a judge gates a release.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from .checks import CheckResult
from .trajectory import Trajectory

JudgeFn = Callable[[str], str]  # prompt -> raw model reply

RUBRIC_PROMPT = """You are grading an AI agent's answer.
Task given to the agent:
{task}

Agent's final answer:
{answer}

Rubric (all must hold for a pass):
{rubric}

Reply with JSON only: {{"pass": true|false, "reason": "<one sentence>"}}"""

PAIRWISE_PROMPT = """Which answer better satisfies the rubric?
Rubric: {rubric}
Task: {task}

Answer A:
{a}

Answer B:
{b}

Reply with JSON only: {{"winner": "A"|"B"|"tie"}}"""


def _parse(reply: str) -> dict | None:
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class RubricJudge:
    def __init__(self, fn: JudgeFn):
        self.fn = fn

    def grade(self, t: Trajectory, rubric: str, task: str = "") -> CheckResult:
        reply = self.fn(RUBRIC_PROMPT.format(task=task, answer=t.final, rubric=rubric))
        verdict = _parse(reply)
        if not verdict or not isinstance(verdict.get("pass"), bool):
            return CheckResult("judge", False, f"unparseable judge reply: {reply[:80]!r}")
        return CheckResult("judge", verdict["pass"], str(verdict.get("reason", "")))


def pairwise(fn: JudgeFn, task: str, a: str, b: str, rubric: str) -> str:
    """Return "A", "B" or "tie", requiring the verdict to survive an order swap."""
    first = (_parse(fn(PAIRWISE_PROMPT.format(rubric=rubric, task=task, a=a, b=b))) or {}).get("winner")
    second = (_parse(fn(PAIRWISE_PROMPT.format(rubric=rubric, task=task, a=b, b=a))) or {}).get("winner")
    swapped = {"A": "B", "B": "A"}.get(second, second)
    return first if first in ("A", "B") and first == swapped else "tie"


def judge_agreement(judge_labels: list[bool], human_labels: list[bool]) -> dict[str, float]:
    n = len(human_labels)
    agree = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    pj, ph = sum(judge_labels) / n, sum(human_labels) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    return {"agreement": agree, "kappa": kappa}


def keyword_judge(required: dict[str, list[str]]) -> JudgeFn:
    """Offline stand-in for an LLM judge: passes if every rubric line's keywords appear.

    ``required`` maps a rubric line to keywords; used in tests and the offline demo.
    """
    def fn(prompt: str) -> str:
        answer = prompt.split("Agent's final answer:", 1)[-1].split("Rubric", 1)[0].lower()
        missing = [line for line, kws in required.items() if line in prompt and not any(k in answer for k in kws)]
        return json.dumps({"pass": not missing, "reason": "missing: " + "; ".join(missing) if missing else "ok"})
    return fn
