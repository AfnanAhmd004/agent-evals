"""Markdown reports for humans (PR comments, CI summaries)."""
from __future__ import annotations

from .gate import GateResult
from .runner import RunReport


def run_markdown(r: RunReport) -> str:
    s = r.summary()
    k = s["trials_per_case"]
    lo, hi = s["pass_rate_ci95"]
    lines = [
        f"### {s['agent']} on `{s['suite']}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases × trials | {s['cases']} × {k} |",
        f"| pass rate (95% CI) | {s['pass_rate']:.1%} ({lo:.1%}–{hi:.1%}) |",
        f"| pass@{k} (solves it at least once) | {s[f'pass@{k}']:.1%} |",
        f"| pass^{k} (solves it every time) | {s[f'pass^{k}']:.1%} |",
        f"| critical cases, pass^{k} | {s['critical_pass^k']:.1%} |",
        f"| latency p50 / p95 | {s['latency_p50_s'] * 1e3:.0f} ms / {s['latency_p95_s'] * 1e3:.0f} ms |",
        f"| steps / tool calls per task | {s['mean_steps']:.1f} / {s['mean_tool_calls']:.1f} |",
        f"| cost per task | ${s['cost_usd_per_task']:.4f} |",
        f"| agent errors | {s['errors']} |",
        "",
        "By tag: " + ", ".join(f"`{t}` {v:.0%}" for t, v in s["by_tag"].items()),
    ]
    if s["top_failing_checks"]:
        lines.append("Most-failed checks: " + ", ".join(f"`{c}` ×{n}" for c, n in s["top_failing_checks"].items()))
    return "\n".join(lines)


def gate_markdown(g: GateResult) -> str:
    head = "✅ **gate passed**" if g.passed else "❌ **gate failed**"
    lines = [head, "", f"Mean per-case change {g.diff:+.1%} (95% CI {g.ci95[0]:+.1%} to {g.ci95[1]:+.1%})"]
    lines += [f"- {r}" for r in g.reasons]
    if g.fixed:
        lines.append(f"Fixed: {', '.join(g.fixed)}")
    if g.regressed:
        lines.append(f"Regressed: {', '.join(g.regressed)}")
    return "\n".join(lines)
