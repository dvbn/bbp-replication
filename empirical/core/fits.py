"""Historical safeguarded Newton roots for the current PPML/GPML application.

The original Gamma pilot, fixed merit, admission/refinement, C2 then C1
starts, joint/profiled equations and fallback rules are retained. The later
unreported score orders 0.5, 1.5 and 2.5 are omitted: they occur after C1
and do not supply its pilot, neighbour or PPML start. This code is Newton
with the actual, potentially nonsymmetric Jacobian; it is not iOLS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from wz_fe_sim.estimators import continuation as cont
from wz_fe_sim.estimators.factorial import (Attempt, RootRecord, _accept_joint, _attempt, _continue_joint,
                                            build_metric)
from wz_fe_sim.estimators.newton import (METRIC_SCOPE_PILOT_PRELIM, SolverControl, admit_with_metric,
                                         newton_solve)

TAUS = (1.0, 2.0)
METRIC_ONLY_GATES = {"metric_residual", "metric_step"}


@dataclass
class ExpFits:
    merit_source: str = ""
    metric: dict = field(default_factory=dict)
    pilot: object = None  # Attempt: preliminary objective-only pilot
    admission_initial: dict | None = None
    refinement: object = None  # Attempt or None
    admission_final: dict | None = None
    pilot_admitted: bool = False
    roots: dict = field(default_factory=dict)  # ("C"|"J", tau) -> RootRecord
    solve_order: tuple = ()
    neighbor: dict = field(default_factory=dict)
    continuation: dict = field(default_factory=dict)


def _admission(cert):
    if cert is None:
        return None
    return {"ok": bool(cert.ok), "metric_scope": cert.metric_scope, "resid_scaled": float(cert.resid_scaled),
            "newton_corr_norm": float(cert.newton_corr_norm), "rcond": float(cert.rcond),
            "metric_residual": float(cert.metric_residual), "metric_step": float(cert.metric_step),
            "failed_gates": list(cert.failed_gates), "message": cert.message}


def fit_exponential(panel, cfg, additional_starts=()):
    ctl = SolverControl.from_config(cfg)
    cc = cont.ContinuationControl.from_config(cfg)
    tol_root = float(cfg["tolerances"]["tol_root"])
    run_all = bool(cfg["run_all_starts"])
    order = tuple(float(t) for t in cfg["cancel_solve_order"] if float(t) in TAUS)
    neighbor = {float(k): (None if v is None else float(v))
                for k, v in cfg["continuation_neighbor"].items() if float(k) in TAUS}
    if set(order) != set(TAUS) or any(v is not None and v not in order for v in neighbor.values()):
        raise ValueError("WZ configuration must cover the requested PPML/GPML orders and its neighbour starts")
    out = ExpFits(solve_order=order, neighbor=neighbor, continuation=cc.describe())
    fallback_starts = [("application_zero", np.zeros(panel.d)), *additional_starts]

    # ---- 0. Gamma pilot, fixed metric, admission, metric-only refinement ----
    pilot = newton_solve(panel.evaluator("C", 2.0), np.zeros(panel.d), ctl, objective=panel.objective(2),
                         start_label="zero", metric_scope=METRIC_SCOPE_PILOT_PRELIM)
    out.pilot = _attempt("zero", pilot, phase="pilot", target_tau=2.0, solved_tau=2.0)
    merit, out.merit_source = build_metric(panel, pilot)
    out.metric = merit.describe() if merit is not None else {"metric_source": out.merit_source}
    admission = admit_with_metric(pilot.certificate, merit, ctl) if (pilot.ok and merit is not None) else None
    out.admission_initial = _admission(admission)
    if admission is not None and not admission.ok and admission.failed_gates \
            and set(admission.failed_gates) <= METRIC_ONLY_GATES:
        ref = newton_solve(panel.evaluator("C", 2.0), pilot.theta, ctl, merit=merit,
                           start_label="gamma_metric_refinement")
        out.refinement = _attempt("gamma_metric_refinement", ref, phase="pilot", target_tau=2.0, solved_tau=2.0)
        pilot = ref
        admission = admit_with_metric(ref.certificate, merit, ctl) if ref.ok else None
        out.admission_final = _admission(admission)
    else:
        out.admission_final = out.admission_initial
    out.pilot_admitted = bool(merit is not None and pilot.ok and admission is not None and admission.ok)
    if not out.pilot_admitted:
        msg = f"no admissible start: gamma pilot {pilot.status} / not admitted; merit {out.merit_source}"
        for tau in order:
            out.roots[("C", tau)] = RootRecord(tau, "C", message=msg, continuation_status=cont.NOT_APPLICABLE)
            if tau != 1.0:
                out.roots[("J", tau)] = RootRecord(tau, "J", message=msg,
                                                   continuation_status=cont.NO_ADMISSIBLE_START)
        return out

    # ---- 1. cancelled roots --------------------------------------------------
    for tau in order:
        rec = RootRecord(tau, "C", continuation_status=cont.NOT_APPLICABLE)
        starts = [("gamma_pilot", pilot.theta)]
        nb = neighbor[tau]
        if nb is not None and _ok(out.roots.get(("C", nb))):
            starts.append((f"neighbor_tau{nb:g}", out.roots[("C", nb)].theta))
        if tau != 1.0 and _ok(out.roots.get(("C", 1.0))):
            starts.append(("ppml_root", out.roots[("C", 1.0)].theta))
        evaluate = panel.evaluator("C", tau)
        executed, solved, stopped = [], [], False
        for label, th in starts:
            if stopped:
                rec.alternatives.append(_attempt_stub(label, "not_attempted", tau))
                continue
            dup = next((lab for lab, t2 in executed if np.array_equal(t2, th)), None)
            if dup is not None:
                rec.alternatives.append(_attempt_stub(label, f"reused:{dup}", tau, "reused"))
                continue
            res = newton_solve(evaluate, th, ctl, merit=merit, start_label=label)
            executed.append((label, th))
            solved.append(res)
            rec.attempts += 1
            rec.attempts_ok += int(res.ok)
            rec.newton_calls += 1
            rec.alternatives.append(_attempt(label, res, phase="cancelled", target_tau=tau, solved_tau=tau))
            stopped = res.ok and not run_all
        accepted = next((r for r in solved if r.ok), None)
        reported = accepted if accepted is not None else (solved[0] if solved else None)
        if reported is not None:
            rec.elapsed_root_s = reported.elapsed_s
            rec.elapsed_retry_s = float(sum(r.elapsed_s for r in solved if r is not reported))
            rec.iterations, rec.backtracks = reported.iterations, reported.backtracks
            rec.terminal, rec.start_used = reported.terminal, reported.start_label
            rec.certified_full_step = bool(reported.certified_full_steps)
            rec.metric_scope = reported.metric_scope
        if accepted is not None:
            rec.status, rec.theta, rec.cert = "ok", accepted.theta, accepted.certificate
            # full coefficient vector (scaled coordinates), not only theta[0]
            rec.ambiguity_max_diff = max(float(np.max(np.abs(r.theta - accepted.theta))) for r in solved if r.ok)
            rec.ambiguous = bool(rec.ambiguity_max_diff > tol_root)
        elif reported is not None:
            rec.status, rec.message = reported.status, f"{reported.start_label}: {reported.message}"
        out.roots[("C", tau)] = rec

        if rec.status != "ok":
            _application_fallback(rec, evaluate, fallback_starts, ctl, merit)

    # ---- 2. joint roots, direct attempt ----------------------------------------
    for tau in order:
        if tau == 1.0:
            continue
        rec = RootRecord(tau, "J")
        c = out.roots[("C", tau)]
        label, th = (f"cancelled_tau{tau:g}", c.theta) if c.status == "ok" else ("gamma_pilot", pilot.theta)
        res = newton_solve(panel.evaluator("J", tau), th, ctl, merit=merit, start_label=label)
        rec.attempts, rec.attempts_ok, rec.newton_calls = 1, int(res.ok), 1
        rec.alternatives.append(_attempt(label, res, phase="direct", target_tau=tau, solved_tau=tau))
        rec.status, rec.iterations, rec.backtracks = res.status, res.iterations, res.backtracks
        rec.start_used, rec.message, rec.elapsed_root_s = label, res.message, res.elapsed_s
        rec.terminal, rec.metric_scope = res.terminal, res.metric_scope
        rec.certified_full_step = bool(res.certified_full_steps)
        if res.ok:
            _accept_joint(rec, res)
            rec.ambiguity_max_diff = 0.0
            rec.continuation_status = cont.NOT_NEEDED
        else:
            rec.continuation_status = cont.DISABLED if not cc.enabled else ""
        out.roots[("J", tau)] = rec

    # ---- 2b. bounded tau continuation for failed joint targets ----------------
    if cc.enabled:
        pool = {float(t): (np.array(r.theta, copy=True), f"J{t:g}_direct")
                for (k, t), r in out.roots.items() if k == "J" and r.status == "ok"}
        if _ok(out.roots.get(("C", 1.0))):
            pool[1.0] = (np.array(out.roots[("C", 1.0)].theta, copy=True), "C1_identity_J1")
        for tau in order:
            rec = out.roots.get(("J", tau))
            if tau == 1.0 or rec is None or rec.status == "ok" or rec.continuation_status:
                continue
            _continue_joint(rec, tau, pool, cc, lambda t, th0, lab: newton_solve(
                panel.evaluator("J", t), th0, ctl, merit=merit, start_label=lab))
    for tau in order:
        rec = out.roots.get(("J", tau))
        if rec is not None and rec.status != "ok":
            _application_fallback(rec, panel.evaluator("J", tau), fallback_starts, ctl, merit)
    return out


def _application_fallback(rec, evaluate, starts, ctl, merit):
    """Prespecified extra starts only after failure; never change root thresholds."""
    for label, theta in starts:
        res = newton_solve(evaluate, theta, ctl, merit=merit, start_label=label)
        rec.attempts += 1
        rec.newton_calls += 1
        rec.attempts_ok += int(res.ok)
        rec.alternatives.append(_attempt(label, res, phase="application_fallback",
                                        target_tau=rec.tau, solved_tau=rec.tau))
        rec.elapsed_retry_s += res.elapsed_s
        if res.ok:
            rec.status, rec.theta, rec.cert = "ok", res.theta, res.certificate
            rec.start_used, rec.iterations, rec.backtracks = label, res.iterations, res.backtracks
            rec.terminal, rec.metric_scope = res.terminal, res.metric_scope
            rec.certified_full_step = bool(res.certified_full_steps)
            rec.message = "Certified from an application-only fallback; previous failures retained."
            rec.ambiguity_max_diff = 0.0
            break


def _ok(rec):
    return rec is not None and rec.status == "ok"


def _attempt_stub(label, status, tau, disposition="not_attempted"):
    msg = "an earlier start was accepted and run_all_starts is false" if disposition == "not_attempted" else ""
    return Attempt(label, status, disposition=disposition, phase="cancelled", target_tau=tau, solved_tau=tau,
                   message=msg)


def identity_check_tau1(panel, theta):
    """At tau = 1 the cancelled and profiled joint equations coincide exactly."""
    c = panel.evaluator("C", 1.0)(theta, True)
    j = panel.evaluator("J", 1.0)(theta, True)
    return {"max_abs_dS": float(np.max(np.abs(c.S - j.S))), "max_abs_dH": float(np.max(np.abs(c.H - j.H))),
            "max_abs_dscale": float(np.max(np.abs(c.scale - j.scale)))}


def joint_country_effects(panel, theta, tau, sample, scale):
    """Recovered joint country log multipliers in original X coordinates:
    alpha_g = log(a_g) - Xbar_g' beta, so log E[y_i] = alpha_g + X_i' beta."""
    beta = theta / scale
    nc = int(sample.country.max() + 1)
    log_a = np.full(nc, np.nan)
    for c, st in zip(panel.cohorts, panel.states(theta, tau)):
        g = sample.country[c.rows[:, 0]]
        with np.errstate(divide="ignore"):
            log_a[g] = np.log(st.a)
    cnt = np.bincount(sample.country, minlength=nc)
    xbar = np.stack([np.bincount(sample.country, weights=col, minlength=nc) for col in sample.X.T], 1) / cnt[:, None]
    return log_a - xbar @ beta, log_a


@dataclass
class OLSFit:
    c: float
    theta: np.ndarray
    resid: np.ndarray
    scores: np.ndarray  # (n, p) B_i e_i
    hinv_solve: object  # callable: H^{-1} applied to columns


def fit_ols(sample, B, c):
    """Within-country OLS of log(y + c) on B (country FE absorbed exactly)."""
    ylog = np.log(sample.y + c)
    nc = int(sample.country.max() + 1)
    cnt = np.bincount(sample.country, minlength=nc)
    yc = ylog - (np.bincount(sample.country, weights=ylog, minlength=nc) / cnt)[sample.country]
    H = B.T @ B
    fac = cho_factor(H, lower=True)  # raises LinAlgError if B is rank deficient
    theta = cho_solve(fac, B.T @ yc)
    e = yc - B @ theta
    return OLSFit(float(c), theta, e, B * e[:, None], lambda m: cho_solve(fac, m))
