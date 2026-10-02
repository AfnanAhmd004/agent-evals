import json
import os

import numpy as np
import pytest

from agentevals import (
    RubricJudge,
    RunReport,
    ToolCall,
    Trajectory,
    compare,
    judge_agreement,
    pairwise,
    pass_at_k,
    pass_hat_k,
    run_check,
    run_suite,
    wilson,
)
from agentevals.cli import main as cli

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def traj(final="", *calls):
    return Trajectory(final, [ToolCall(n, a) for n, a in calls], steps=len(calls) + 1)


def test_outcome_checks():
    t = traj('{"robot": {"id": "arm-07", "ok": false}}')
    assert run_check(t, {"type": "json_field", "path": "robot.id", "equals": "arm-07"}).passed
    assert not run_check(t, {"type": "json_field", "path": "robot.missing", "equals": 1}).passed
    assert run_check(traj("Ticket WO-1234 opened"), {"type": "regex", "pattern": r"WO-\d+"}).passed
    assert run_check(traj("STOPPED"), {"type": "contains", "value": "stopped"}).passed


def test_process_checks():
    t = traj("", ("get_status", {"robot": "ARM-07"}), ("log", {}), ("lookup_error", {"code": "E-217"}))
    assert run_check(t, {"type": "tool_order", "order": ["get_status", "lookup_error"]}).passed
    assert not run_check(t, {"type": "tool_order", "order": ["lookup_error", "get_status"]}).passed
    assert run_check(t, {"type": "tool_called", "name": "get_status", "args": {"robot": "arm-07"}}).passed  # case-insensitive
    assert not run_check(t, {"type": "tool_called", "name": "get_status", "args": {"robot": "arm-03"}}).passed
    assert not run_check(t, {"type": "max_tool_calls", "value": 2}).passed
    with pytest.raises(ValueError):
        run_check(t, {"type": "nope"})


def test_pass_at_k_and_pass_hat_k():
    assert pass_at_k(10, 0, 3) == 0 and pass_at_k(10, 10, 3) == 1
    assert pass_at_k(5, 1, 1) == pytest.approx(0.2)
    # an agent that is right 50% of the time "solves" most tasks at k=5 but is rarely reliable
    assert pass_at_k(10, 5, 5) > 0.99 and pass_hat_k(10, 5, 5) < 0.01


def test_wilson_interval():
    lo, hi = wilson(0, 20)
    assert lo == 0 and 0.1 < hi < 0.2
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi


def test_agent_crash_is_a_failed_trial_not_a_crashed_eval():
    def bad(_):
        raise RuntimeError("model API 500")

    suite = {"name": "s", "cases": [{"id": "a", "input": "x", "checks": [{"type": "contains", "value": "x"}]}]}
    r = run_suite(bad, suite, trials=3)
    assert r.cases[0].n_pass == 0 and r.summary()["errors"] == 3


def test_report_roundtrip(tmp_path):
    suite = {"name": "s", "cases": [{"id": "a", "input": "hi", "tags": ["t"], "checks": [{"type": "contains", "value": "hi"}]}]}
    r = run_suite(lambda s: Trajectory(s.upper(), steps=1), suite, trials=2)
    r.save(tmp_path / "r.json")
    r2 = RunReport.load(tmp_path / "r.json")
    assert r2.summary()["pass_rate"] == 1.0 and r2.cases[0].trials[0].trajectory.final == "HI"


def test_rubric_judge_fails_closed_on_garbage():
    assert not RubricJudge(lambda p: "sure, looks great").grade(traj("x"), rubric="r").passed
    assert RubricJudge(lambda p: '{"pass": true, "reason": "ok"}').grade(traj("x"), rubric="r").passed


def test_pairwise_cancels_position_bias():
    always_first = lambda p: '{"winner": "A"}'
    assert pairwise(always_first, "t", "x", "y", "r") == "tie"
    def prefers_long(p):
        a = p.split("Answer A:")[1].split("Answer B:")[0]
        b = p.split("Answer B:")[1].split("Reply with")[0]
        return '{"winner": "%s"}' % ("A" if len(a) > len(b) else "B")

    assert pairwise(prefers_long, "t", "a much longer answer", "short", "r") == "A"


def test_judge_agreement_kappa():
    human = [True, True, False, False] * 10
    assert judge_agreement(human, human)["kappa"] == pytest.approx(1.0)
    assert judge_agreement([True] * 40, human)["kappa"] == pytest.approx(0.0)


def _report(scores, critical=()):
    suite = {"name": "s", "cases": [{"id": str(i), "input": str(i), "critical": str(i) in critical,
                                     "checks": [{"type": "contains", "value": "ok"}]} for i in range(len(scores))]}
    counters = {str(i): 0 for i in range(len(scores))}

    def agent(inp):
        counters[inp] += 1
        return Trajectory("ok" if (counters[inp] - 1) < scores[int(inp)] * 4 else "no", steps=1)

    return run_suite(agent, suite, trials=4, workers=1)


def test_gate_passes_identical_and_catches_regressions():
    base = _report([1.0] * 20, critical={"0"})
    assert compare(base, _report([1.0] * 20, critical={"0"})).passed
    worse = compare(base, _report([0.5] * 20, critical={"0"}))
    assert not worse.passed and any("significant" in r for r in worse.reasons)
    # better on average, but one critical case breaks: still a fail
    mixed = compare(_report([0.5] * 19 + [1.0], critical={"19"}), _report([1.0] * 19 + [0.75], critical={"19"}))
    assert mixed.diff > 0 and not mixed.passed and mixed.regressed == ["19"]


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    suite = os.path.join(ROOT, "suites", "robot_ops.json")
    v1, v2 = str(tmp_path / "v1.json"), str(tmp_path / "v2.json")
    assert cli(["run", suite, "--agent", "examples.ops_agent:agent_v1", "--trials", "4", "--out", v1]) == 0
    assert cli(["run", suite, "--agent", "examples.ops_agent:agent_v2", "--trials", "8", "--out", v2]) == 0
    assert cli(["compare", v1, v1]) == 0
    assert cli(["compare", v1, v2]) == 1  # v2's safety regression blocks the release
    assert json.load(open(v2))["trials"] == 8
