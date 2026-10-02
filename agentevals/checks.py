"""Deterministic checks on a trajectory. Each check is declared in the suite as JSON.

Outcome checks look at the final answer; process checks look at what the agent
did (which tools, with which arguments, in what order, how many steps). Process
checks catch agents that get the right answer the wrong way, e.g. by skipping a
mandatory safety lookup.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .trajectory import Trajectory


@dataclass
class CheckResult:
    check: str
    passed: bool
    detail: str = ""


def _subset(expected: dict, actual: dict) -> bool:
    for k, v in expected.items():
        if k not in actual:
            return False
        a = actual[k]
        if isinstance(v, str) and isinstance(a, str):
            if v.lower() != a.lower():
                return False
        elif v != a:
            return False
    return True


def contains(t: Trajectory, value: str, case_sensitive: bool = False) -> CheckResult:
    hay, needle = (t.final, value) if case_sensitive else (t.final.lower(), value.lower())
    return CheckResult(f"contains:{value}", needle in hay)


def not_contains(t: Trajectory, value: str) -> CheckResult:
    return CheckResult(f"not_contains:{value}", value.lower() not in t.final.lower())


def regex(t: Trajectory, pattern: str) -> CheckResult:
    return CheckResult(f"regex:{pattern}", re.search(pattern, t.final) is not None)


def json_field(t: Trajectory, path: str, equals: Any) -> CheckResult:
    """The final answer is JSON and ``path`` (dot-separated) equals a value."""
    try:
        node = json.loads(t.final)
        for part in path.split("."):
            node = node[int(part)] if isinstance(node, list) else node[part]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return CheckResult(f"json_field:{path}", False, f"not found: {exc!r}")
    return CheckResult(f"json_field:{path}", node == equals, f"got {node!r}")


def tool_called(t: Trajectory, name: str, args: dict | None = None, times: int | None = None) -> CheckResult:
    hits = [c for c in t.tool_calls if c.name == name and (args is None or _subset(args, c.args))]
    ok = len(hits) == times if times is not None else bool(hits)
    return CheckResult(f"tool_called:{name}", ok, f"{len(hits)} matching call(s)")


def tool_not_called(t: Trajectory, name: str) -> CheckResult:
    n = sum(c.name == name for c in t.tool_calls)
    return CheckResult(f"tool_not_called:{name}", n == 0, f"{n} call(s)")


def tool_order(t: Trajectory, order: list[str]) -> CheckResult:
    """The tools appear in this relative order (other calls may be interleaved)."""
    it = iter(c.name for c in t.tool_calls)
    ok = all(any(n == want for n in it) for want in order)
    return CheckResult(f"tool_order:{'>'.join(order)}", ok)


def max_steps(t: Trajectory, value: int) -> CheckResult:
    return CheckResult(f"max_steps:{value}", t.steps <= value, f"{t.steps} steps")


def max_tool_calls(t: Trajectory, value: int) -> CheckResult:
    return CheckResult(f"max_tool_calls:{value}", len(t.tool_calls) <= value, f"{len(t.tool_calls)} calls")


def no_error(t: Trajectory) -> CheckResult:
    return CheckResult("no_error", t.error is None, t.error or "")


CHECKS: dict[str, Callable[..., CheckResult]] = {
    "contains": contains,
    "not_contains": not_contains,
    "regex": regex,
    "json_field": json_field,
    "tool_called": tool_called,
    "tool_not_called": tool_not_called,
    "tool_order": tool_order,
    "max_steps": max_steps,
    "max_tool_calls": max_tool_calls,
    "no_error": no_error,
}


def run_check(t: Trajectory, spec: dict, judge=None, task: str = "") -> CheckResult:
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "judge":
        if judge is None:
            return CheckResult("judge", False, "no judge configured")
        return judge.grade(t, task=task, **spec)
    if kind not in CHECKS:
        raise ValueError(f"unknown check type {kind!r}")
    return CHECKS[kind](t, **spec)
