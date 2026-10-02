"""Release gate: compare a candidate run against a baseline, case by case."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import paired_bootstrap
from .runner import RunReport


@dataclass
class GateResult:
    passed: bool
    diff: float
    ci95: tuple[float, float]
    reasons: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)


def compare(base: RunReport, cand: RunReport, max_drop: float = 0.02, min_pass_rate: float | None = None) -> GateResult:
    """Fail the candidate if any of these hold:

    * its mean per-case score drops by more than ``max_drop``;
    * the drop is statistically significant (95% paired-bootstrap CI entirely below 0);
    * any ``critical`` case that passed every trial in the baseline no longer does;
    * its overall pass rate is below ``min_pass_rate``.

    Per-case pairing matters: two agents with the same headline pass rate can fail
    completely different cases, and the regressions are what users notice.
    """
    b = {c.id: c for c in base.cases}
    shared = [c for c in cand.cases if c.id in b]
    if not shared:
        raise ValueError("no overlapping case ids between runs")
    bs = np.array([b[c.id].score for c in shared])
    cs = np.array([c.score for c in shared])
    diff, lo, hi = paired_bootstrap(bs, cs)
    regressed = [c.id for c in shared if c.score < b[c.id].score]
    fixed = [c.id for c in shared if c.score > b[c.id].score]
    reasons = []
    if diff < -max_drop:
        reasons.append(f"mean score dropped {-diff:.1%} (> {max_drop:.1%} allowed)")
    if hi < 0:
        reasons.append(f"regression is significant (95% CI {lo:+.1%} to {hi:+.1%})")
    crit = [c.id for c in shared if c.critical and b[c.id].n_pass == len(b[c.id].trials) and c.n_pass < len(c.trials)]
    if crit:
        reasons.append(f"critical case(s) regressed: {', '.join(crit)}")
    if min_pass_rate is not None:
        rate = cand.summary()["pass_rate"]
        if rate < min_pass_rate:
            reasons.append(f"pass rate {rate:.1%} below floor {min_pass_rate:.1%}")
    return GateResult(not reasons, diff, (lo, hi), reasons, regressed, fixed)
