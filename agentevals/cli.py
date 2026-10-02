"""Command line: run a suite, compare two runs, print a report.

    agent-evals run suites/robot_ops.json --agent examples.ops_agent:agent_v2 --trials 5 --out runs/v2.json
    agent-evals compare runs/v1.json runs/v2.json --max-drop 0.02     # exit 1 if the gate fails
    agent-evals report runs/v2.json
"""
from __future__ import annotations

import argparse
import importlib
import sys

from .gate import compare
from .report import gate_markdown, run_markdown
from .runner import RunReport, load_suite, run_suite


def _load_agent(ref: str):
    module, _, attr = ref.partition(":")
    sys.path.insert(0, ".")
    return getattr(importlib.import_module(module), attr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent-evals")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("suite")
    r.add_argument("--agent", required=True, help="module:function returning a Trajectory")
    r.add_argument("--trials", type=int, default=3)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--max-drop", type=float, default=0.02)
    c.add_argument("--min-pass-rate", type=float)
    p = sub.add_parser("report")
    p.add_argument("run")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        report = run_suite(_load_agent(args.agent), load_suite(args.suite), trials=args.trials,
                           workers=args.workers, agent_name=args.agent)
        report.save(args.out)
        print(run_markdown(report))
        return 0
    if args.cmd == "compare":
        g = compare(RunReport.load(args.baseline), RunReport.load(args.candidate), args.max_drop, args.min_pass_rate)
        print(gate_markdown(g))
        return 0 if g.passed else 1
    print(run_markdown(RunReport.load(args.run)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
