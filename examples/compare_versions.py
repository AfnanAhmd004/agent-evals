"""Evaluate agent v1 and v2 on the robot-ops suite and run the release gate.

    python examples/compare_versions.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentevals import RubricJudge, compare, gate_markdown, keyword_judge, load_suite, run_markdown, run_suite
from examples.ops_agent import agent_v1, agent_v2

# Offline stand-in for an LLM judge. Swap in any `prompt -> reply` function, e.g. a Claude call.
JUDGE = RubricJudge(keyword_judge({
    "mentions why navigation or localisation is affected": ["navigation", "localisation", "localization"],
    "names the component to inspect": ["gearbox"],
}))


def main() -> None:
    suite = load_suite(os.path.join(os.path.dirname(__file__), "..", "suites", "robot_ops.json"))
    os.makedirs("runs", exist_ok=True)
    reports = {}
    for agent in (agent_v1, agent_v2):
        r = run_suite(agent, suite, trials=8, judge=JUDGE, workers=1)  # workers=1: reproducible
        r.save(f"runs/{agent.__name__}.json")
        reports[agent.__name__] = r
        print(run_markdown(r), "\n")
    g = compare(reports["agent_v1"], reports["agent_v2"], max_drop=0.02)
    print(gate_markdown(g))


if __name__ == "__main__":
    main()
