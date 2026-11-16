"""Author-approved bounded tau continuation for FAILED joint targets only.

Joint-root continuation protocol:
  * It is an additional START, run only after the prescribed direct J attempt
    of a target failed.  It never refits or replaces an accepted direct root and
    never changes the target equation, tau, Newton derivative, merit/metric or
    acceptance thresholds.
  * Seed: from a FROZEN snapshot of certified direct J roots plus C(1) = J(1)
    of the same draw-cell; the nearest seed in |tau - target| wins, ties go to
    the smaller tau.  The cancelled Gamma pilot is not a J(2) root.  Later
    continuation successes and intermediate points are never added to the pool.
  * Path: tau_trial = tau_current + direction * min(step, |target - tau_current|);
    the terminal trial is exactly ``target``.  Each trial is one call of the
    supplied ``solve`` (newton_solve on the original profiled joint equation at
    tau_trial, from the last CERTIFIED path point, same fixed merit/metric and
    the same shared acceptance routine).  Success advances (and doubles the step,
    capped, after ``growth_after_successes`` consecutive successes; the counter
    then restarts); failure keeps tau and coefficients, resets the success count
    and contracts the step.  A failed terminal iterate is never a warm start.
  * Stops at the certified exact target, when the step would fall below
    ``step_min``, when ``max_calls_per_target`` calls have been made, or when no
    numerical tau progress is possible.  Exhaustion is a failure of this bounded
    search, not evidence that no root exists; success is no uniqueness or
    branch-identity claim.
Every executed call is returned in chronological order (one record per call).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TAU_BOUNDS = (0.0, 3.0)

# continuation_status vocabulary (per J target)
NOT_NEEDED = "not_needed"  # direct attempt accepted: no continuation call
DISABLED = "disabled"  # direct attempt failed, continuation switched off in config
NO_SEED = "no_certified_seed"
ENDPOINT_OK = "endpoint_certified"
MIN_STEP = "min_step_reached"
BUDGET = "budget_exhausted"
NO_PROGRESS = "no_numerical_progress"
NOT_APPLICABLE = "not_applicable"  # cancelled roots
NO_ADMISSIBLE_START = "no_admissible_start"  # Gamma pilot failed / not admitted: no J attempt at all
CONTINUATION_STATUSES = (NOT_NEEDED, DISABLED, NO_SEED, ENDPOINT_OK, MIN_STEP, BUDGET, NO_PROGRESS,
                         NOT_APPLICABLE, NO_ADMISSIBLE_START)

# configuration keys / validation live in config.py (standard library only)
from ..config import JOINT_CONTINUATION_KEYS as CONFIG_KEYS  # noqa: E402,F401
from ..config import JOINT_CONTINUATION_SEED_POLICY as SEED_POLICY  # noqa: E402
from ..config import validate_joint_continuation as validate_config  # noqa: E402,F401


@dataclass(frozen=True)
class ContinuationControl:
    enabled: bool = True
    step_initial: float = 0.5
    step_max: float = 0.5
    step_min: float = 1.0 / 64.0
    contraction: float = 0.5
    growth: float = 2.0
    growth_after_successes: int = 2
    max_calls_per_target: int = 32
    seed_policy: str = SEED_POLICY

    @classmethod
    def from_config(cls, cfg):
        jc = cfg["joint_continuation"]
        return cls(bool(jc["enabled"]), float(jc["step_initial"]), float(jc["step_max"]),
                   float(jc["step_min"]), float(jc["contraction"]), float(jc["growth"]),
                   int(jc["growth_after_successes"]), int(jc["max_calls_per_target"]), str(jc["seed_policy"]))

    def describe(self):
        return dict(vars(self))


def select_seed(target, pool):
    """Nearest seed in |tau - target|, ties to the smaller tau; None if empty.

    ``pool`` maps seed tau -> (theta, label).  Order of ``pool`` is irrelevant."""
    cands = sorted((abs(float(t) - float(target)), float(t)) for t in pool if float(t) != float(target))
    if not cands:
        return None
    tau = cands[0][1]
    theta, label = pool[tau]
    return tau, theta, label


@dataclass
class PathCall:
    """One executed Newton call of a continuation path (chronological)."""

    phase: str  # "intermediate" or "endpoint"
    target_tau: float
    solved_tau: float
    source_tau: float  # tau of the certified point used as the start
    anchor: str  # seed label of this path
    start_label: str
    trial_step: float
    step_reductions: int  # contractions so far on this path (before this call)
    result: object  # SolveResult


@dataclass
class ContinuationOutcome:
    status: str
    seed_tau: float = np.nan
    seed_label: str = ""
    last_certified_tau: float = np.nan
    calls: list = field(default_factory=list)
    endpoint: object = None  # accepted SolveResult at exactly target (status ENDPOINT_OK only)
    step_reductions: int = 0
    message: str = ""


def continue_to_target(solve, target, seed, cc):
    """Run one deterministic bounded path from ``seed`` = (tau, theta, label).

    ``solve(tau, theta0, start_label)`` must run the unchanged Newton solve on
    the original joint equation at ``tau`` and return its SolveResult."""
    target = float(target)
    seed_tau, seed_theta, seed_label = float(seed[0]), np.array(seed[1], dtype=float), seed[2]
    lo, hi = min(seed_tau, target), max(seed_tau, target)
    if not (TAU_BOUNDS[0] <= lo and hi <= TAU_BOUNDS[1]):
        raise ValueError("seed and target must lie in [0, 3]")
    out = ContinuationOutcome("", seed_tau, seed_label, seed_tau)
    cur_tau, cur_theta = seed_tau, seed_theta
    step, run_ok, reductions = float(cc.step_initial), 0, 0
    while True:
        if step < cc.step_min:
            out.status = MIN_STEP
            break
        if len(out.calls) >= cc.max_calls_per_target:
            out.status = BUDGET
            break
        dist = target - cur_tau
        h = min(step, abs(dist))
        trial = target if h == abs(dist) else cur_tau + math.copysign(h, dist)
        if trial == cur_tau or h == 0.0:
            out.status = NO_PROGRESS
            break
        if not (lo <= trial <= hi and TAU_BOUNDS[0] <= trial <= TAU_BOUNDS[1]):
            raise RuntimeError(f"continuation trial {trial!r} outside [{lo}, {hi}]")  # never expected
        phase = "endpoint" if trial == target else "intermediate"
        label = f"joint_continuation[{seed_label}]:tau{cur_tau:g}->tau{trial:g}"
        res = solve(trial, cur_theta, label)
        out.calls.append(PathCall(phase, target, trial, cur_tau, seed_label, label, h, reductions, res))
        if res.ok:
            if phase == "endpoint":
                if trial != target:  # explicit evaluated-tau == requested-tau check
                    raise RuntimeError("endpoint tau differs from the requested tau")
                out.status, out.endpoint, out.last_certified_tau = ENDPOINT_OK, res, trial
                break
            cur_tau, cur_theta = trial, np.array(res.theta, dtype=float)
            out.last_certified_tau = cur_tau
            run_ok += 1
            if run_ok >= cc.growth_after_successes:
                step, run_ok = min(step * cc.growth, cc.step_max), 0
        else:  # keep tau and coefficients; never warm-start from the failed iterate
            run_ok = 0
            step *= cc.contraction
            reductions += 1
    out.step_reductions = reductions
    if out.status != ENDPOINT_OK:
        out.message = (f"continuation {out.status} after {len(out.calls)} call(s) from {seed_label} "
                       f"(last certified tau {out.last_certified_tau:g}); not evidence of non-existence")
    return out


__all__ = ["ContinuationControl", "ContinuationOutcome", "PathCall", "continue_to_target", "select_seed",
           "validate_config", "CONTINUATION_STATUSES", "SEED_POLICY"]
