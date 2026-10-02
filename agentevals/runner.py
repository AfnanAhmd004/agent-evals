"""Run an agent over a suite, several trials per case, and summarise the results."""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .checks import CheckResult, run_check
from .metrics import pass_at_k, pass_hat_k, percentile, wilson
from .trajectory import Trajectory

AgentFn = Callable[[str], Trajectory]


@dataclass
class TrialResult:
    passed: bool
    checks: list[CheckResult]
    trajectory: Trajectory


@dataclass
class CaseResult:
    id: str
    input: str
    tags: list[str]
    critical: bool
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def n_pass(self) -> int:
        return sum(t.passed for t in self.trials)

    @property
    def score(self) -> float:
        return self.n_pass / len(self.trials)


@dataclass
class RunReport:
    suite: str
    agent: str
    trials: int
    cases: list[CaseResult]

    # -- persistence -------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "RunReport":
        with open(path) as f:
            d = json.load(f)
        cases = []
        for c in d["cases"]:
            trials = [TrialResult(t["passed"], [CheckResult(**r) for r in t["checks"]], Trajectory.from_dict(t["trajectory"]))
                      for t in c["trials"]]
            cases.append(CaseResult(c["id"], c["input"], c["tags"], c["critical"], trials))
        return cls(d["suite"], d["agent"], d["trials"], cases)

    # -- metrics -----------------------------------------------------------------
    def summary(self) -> dict:
        k = self.trials
        all_trials = [t for c in self.cases for t in c.trials]
        n_pass = sum(t.passed for t in all_trials)
        lo, hi = wilson(n_pass, len(all_trials))
        trajs = [t.trajectory for t in all_trials]
        by_tag: dict[str, list[float]] = defaultdict(list)
        for c in self.cases:
            for tag in c.tags:
                by_tag[tag].append(c.score)
        failing_checks = Counter(r.check.split(":")[0] for t in all_trials for r in t.checks if not r.passed)
        return {
            "suite": self.suite,
            "agent": self.agent,
            "cases": len(self.cases),
            "trials_per_case": k,
            "pass_rate": n_pass / len(all_trials),
            "pass_rate_ci95": (lo, hi),
            f"pass@{k}": float(np.mean([pass_at_k(k, c.n_pass, k) for c in self.cases])),
            f"pass^{k}": float(np.mean([pass_hat_k(k, c.n_pass, k) for c in self.cases])),
            "critical_pass^k": float(np.mean([pass_hat_k(k, c.n_pass, k) for c in self.cases if c.critical] or [float("nan")])),
            "by_tag": {tag: float(np.mean(v)) for tag, v in sorted(by_tag.items())},
            "latency_p50_s": percentile([t.latency_s for t in trajs], 50),
            "latency_p95_s": percentile([t.latency_s for t in trajs], 95),
            "mean_steps": float(np.mean([t.steps for t in trajs])),
            "mean_tool_calls": float(np.mean([len(t.tool_calls) for t in trajs])),
            "cost_usd_per_task": float(np.mean([t.cost() for t in trajs])),
            "errors": sum(t.error is not None for t in trajs),
            "top_failing_checks": dict(failing_checks.most_common(5)),
        }


def load_suite(path: str) -> dict:
    with open(path) as f:
        suite = json.load(f)
    ids = [c["id"] for c in suite["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in suite")
    return suite


def _run_one(agent: AgentFn, case: dict, judge, timeout_s: float) -> TrialResult:
    t0 = time.perf_counter()
    try:
        traj = agent(case["input"])
    except Exception as exc:  # an agent crash is a failed trial, not a crashed eval
        traj = Trajectory(error=f"{type(exc).__name__}: {exc}")
    if not traj.latency_s:
        traj.latency_s = time.perf_counter() - t0
    if traj.latency_s > timeout_s and traj.error is None:
        traj.error = f"timeout: {traj.latency_s:.1f}s > {timeout_s}s"
    checks = [run_check(traj, spec, judge, task=case["input"]) for spec in case.get("checks", [])]
    if traj.error:
        checks.append(CheckResult("no_error", False, traj.error))
    return TrialResult(all(c.passed for c in checks), checks, traj)


def run_suite(agent: AgentFn, suite: dict, trials: int = 3, workers: int = 8, judge=None,
              timeout_s: float = 120.0, agent_name: str | None = None) -> RunReport:
    cases = [CaseResult(c["id"], c["input"], c.get("tags", []), c.get("critical", False)) for c in suite["cases"]]
    jobs = [(i, c) for i, c in enumerate(suite["cases"]) for _ in range(trials)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda job: (job[0], _run_one(agent, job[1], judge, timeout_s)), jobs))
    for i, r in results:
        cases[i].trials.append(r)
    return RunReport(suite.get("name", "suite"), agent_name or getattr(agent, "__name__", "agent"), trials, cases)
