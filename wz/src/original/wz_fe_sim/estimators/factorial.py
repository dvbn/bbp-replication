"""Estimator factorial for one draw-cell: one cancelled root per configured tau
plus one distinct joint root per configured tau != 1 (7 + 6 on the seven-power grid).

Order (contract Sec. 3, 5):
 0. Gamma pilot: cancelled tau=2 root by convex-objective Newton from zero,
    certified first under the coordinate gates.  Its curvature H_+ (SPD) fixes
    the residual merit W = H_+^{-1} and the mean-scale certificate metric
    A = H_+/n for every later solve in this draw-cell; fallback B'B/n (already
    averaged, A = B'B/n) only if the pilot root exists but its curvature fails
    the Cholesky (SPD) check.  The pilot is admitted as the common start only
    after the same candidate also passes the fixed-metric gates
    (newton.admit_with_metric; no new fit).  If the pilot fails or is not
    admitted there is no admissible first start (all later starts depend on
    accepted roots), so every root is recorded as not_attempted.
 1. Cancelled roots in the configured order (seven-power grid: 2,1,1.5,0.5,0,2.5,3).
    Starts in fixed priority: [gamma pilot root, configured continuation
    neighbour (if already accepted), PPML root C(1) (if accepted)].  Starts are
    processed and RECORDED in that order: an executed solve ("attempted"), a
    start whose vector equals an already executed start ("reused:<label>", no
    new solve), or a start skipped because an earlier one was accepted and
    run_all_starts is false ("not_attempted").  With run_all_starts, every
    distinct start is attempted so that root ambiguity can be detected; the
    FIRST accepted root in priority order is reported.
 2. Joint roots for tau != 1, pass 1 (prescribed, unchanged): one direct start,
    C(tau) if accepted, else the Gamma pilot.  J(1) = C(1) by the exact
    profile identity; no third PPML root is fitted.
    Pass 2 (author-approved, config ``joint_continuation``): only FAILED direct
    J targets, in the configured order, get one bounded tau-continuation path
    (estimators/continuation.py) from a frozen snapshot of certified direct J
    roots plus C(1).  Accepted direct roots are never refitted or replaced; the
    original failed direct attempt and every path call are retained.
 3. Post-processing per accepted root: sandwich, inference check, correction
    (cancelled only); PPML extras at tau=1: WZ bias/covariances and SPJ.
No truth start, oracle nuisance, Fisher/IRLS substitution, ridge or tau swap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..design import operators as ops
from . import continuation as cont
from .correction import manuscript_correction, pair_sandwich, target_row
from .newton import (METRIC_SCOPE_PILOT_PRELIM, METRIC_SCOPE_PPML_SUBFIT, Merit, SolverControl, Terminal,
                     admit_with_metric, newton_solve, SOLVER_VERSION)
from .wz_tau1 import spj_combine, spj_quarters, wz_correction

JOINT_SYMMETRY_TOL = 1e-10  # max|Hj - Hj'| / max|Hj| (contract Sec. 4)


@dataclass
class RootRecord:
    tau: float
    kind: str  # "C" or "J"
    status: str = "not_attempted"
    theta: np.ndarray | None = None
    start_used: str = ""
    attempts: int = 0
    attempts_ok: int = 0
    iterations: int = -1
    backtracks: int = -1
    ambiguous: bool = False
    ambiguity_max_diff: float = np.nan
    # Every start in fixed priority order (attempted / reused / not_attempted);
    # for J: the direct attempt followed by every continuation call in order.
    alternatives: list = field(default_factory=list)
    cert: object = None
    terminal: Terminal = field(default_factory=Terminal)  # reported attempt's last iterate
    certified_full_step: bool = False  # reported attempt ended by a certified full Newton step
    elapsed_root_s: float = np.nan
    elapsed_retry_s: float = 0.0
    jac_asym: float = np.nan
    message: str = ""
    metric_scope: str = ""  # reported attempt's certificate scope
    newton_calls: int = 0  # executed Newton calls for this root (direct + continuation)
    attempts_intermediate: int = 0  # continuation calls at intermediate (non-target) tau
    attempts_intermediate_ok: int = 0
    continuation_status: str = ""
    continuation_seed: str = ""
    continuation_last_tau: float = np.nan


@dataclass
class Attempt:
    """One start of one root: status, cost and last-iterate diagnostics.

    ``beta`` is the accepted coefficient of THIS call at ``solved_tau`` (NaN
    unless status == 'ok'); an intermediate continuation call's beta is never a
    target estimate.  ``terminal.beta`` is where the solver stopped and is
    never an estimate.  ``disposition``: attempted (a real Newton call),
    reused (reference to an earlier executed start with the identical vector;
    no new solve, no cost) or not_attempted."""

    label: str
    status: str
    beta: float = np.nan
    iterations: int = -1
    backtracks: int = -1
    elapsed_s: float = np.nan
    terminal: Terminal = field(default_factory=Terminal)
    message: str = ""
    certified_full_step: bool = False  # ended by a certified full Newton step after Armijo failed
    disposition: str = "attempted"
    phase: str = ""  # pilot / cancelled / direct / intermediate / endpoint
    target_tau: float = np.nan
    solved_tau: float = np.nan
    source_tau: float = np.nan  # continuation: tau of the certified start point
    anchor: str = ""  # continuation: seed label of the path
    trial_step: float = np.nan
    step_reductions: int = -1
    metric_scope: str = ""
    rejected_full_step: dict | None = None
    accepted_cert: dict | None = None  # certificate diagnostics of an accepted call


def _cert_summary(cert):
    if cert is None:
        return None
    return {"resid_scaled": float(cert.resid_scaled), "newton_corr_norm": float(cert.newton_corr_norm),
            "rcond": float(cert.rcond), "metric_residual": float(cert.metric_residual),
            "metric_step": float(cert.metric_step), "metric_scope": cert.metric_scope}


def _attempt(label, res, **extra):
    return Attempt(label, res.status, float(res.theta[0]) if res.ok else np.nan, res.iterations,
                   res.backtracks, res.elapsed_s, res.terminal, res.message, bool(res.certified_full_steps),
                   metric_scope=res.metric_scope, rejected_full_step=res.rejected_full_step,
                   accepted_cert=_cert_summary(res.certificate) if res.ok else None, **extra)


@dataclass
class PostRecord:
    se: float = np.nan
    status_inference: str = "not_attempted"
    resid_target_over_se: float = np.nan
    target_resid: float = np.nan  # |a'S| at the root, reused for other SE versions
    schur_beta: float = np.nan
    schur_beta_over_xx: float = np.nan
    bias: float = np.nan
    term1: float = np.nan
    term2: float = np.nan
    status_correction: str = "not_attempted"
    elapsed_cov_s: float = np.nan
    elapsed_corr_s: float = np.nan
    Hinv: np.ndarray | None = None
    message: str = ""


@dataclass
class CellFit:
    pilot_status: str
    merit_source: str
    elapsed_setup_s: float
    roots: dict  # ("C"|"J", tau) -> RootRecord
    post: dict  # same keys -> PostRecord
    wz: object = None
    wz_status: str = "not_attempted"
    spj_beta: float = np.nan
    spj_status: str = "not_attempted"
    spj_quarters: list = field(default_factory=list)
    elapsed_wz_s: float = np.nan
    elapsed_spj_s: float = np.nan
    n_obs: int = 0
    n_pairs: int = 0
    d: int = 0
    w_range: dict = field(default_factory=dict)  # tau -> (wmin, wmax) at accepted C root
    inference_rel: float = 1e-6
    pilot: Attempt | None = None  # Gamma pilot (cancelled tau=2 from zero, objective Newton)
    spj_metric_scope: str = ""  # SPJ quarter subfits: objective-only coordinate certificate
    pilot_admission: dict = field(default_factory=dict)  # fixed-metric admission of the pilot
    metric: dict = field(default_factory=dict)  # metric source / normalisation / n of this cell
    start_policy: str = ""
    continuation: dict = field(default_factory=dict)  # continuation controls used (provenance)
    # Full solver records alongside the legacy four-field quarter tuples.
    # Their elapsed_s values are contained in elapsed_spj_s, never extra costs.
    spj_results: list = field(default_factory=list)  # (quarter name, SolveResult)


def _cancel_eval(design, y, tau):
    return lambda th, jac: ops.cancelled_eval(design, y, th, tau, jac)


def _joint_eval(design, y, tau):
    return lambda th, jac: ops.joint_eval(design, y, th, tau, jac)


def gamma_pilot(design, y, ctl):
    """Cancelled tau=2 root from zero by Newton with objective backtracking.

    Coordinate certificate only (its metric does not exist yet); fit_cell then
    admits it with the fixed metric built from its own curvature."""
    obj = lambda th: ops.cancelled_objective(design, y, th, 2)  # noqa: E731
    return newton_solve(_cancel_eval(design, y, 2.0), np.zeros(design.d), ctl,
                        objective=obj, start_label="zero", metric_scope=METRIC_SCOPE_PILOT_PRELIM)


def ppml_from_zero(design, y, ctl):
    """Cancelled tau=1 root from zero by convex-objective Newton (SPJ subfits).

    Strict coordinate/rcond certificate; no Gamma pilot exists in the subfit
    coordinate space, so the metric gates are explicitly not applicable."""
    obj = lambda th: ops.cancelled_objective(design, y, th, 1)  # noqa: E731
    return newton_solve(_cancel_eval(design, y, 1.0), np.zeros(design.d), ctl,
                        objective=obj, start_label="zero", metric_scope=METRIC_SCOPE_PPML_SUBFIT)


def build_metric(design, pilot):
    """(Merit or None, merit_source) under the documented rule: Gamma curvature
    if SPD, else the design-Gram fallback; never a ridge or pseudo-inverse."""
    merit = None
    if pilot.ok:
        try:
            merit = Merit.gamma_curvature(pilot.certificate.evaluation.H, design.n_obs)
        except np.linalg.LinAlgError:
            merit = None
    if merit is None:
        try:
            merit = Merit.design_gram_fallback(design.gram() / design.n_obs, design.n_obs)
        except np.linalg.LinAlgError:
            return None, "unavailable_design_rank_deficient"
    return merit, merit.source


def _tau_key(tau):
    return float(tau)


def fit_cell(cell, cfg, design=None):
    """Run the full factorial on one CellData; returns CellFit (never raises on
    numerical failure: failures are recorded in statuses)."""
    from ..versions import start_policy_version

    ctl = SolverControl.from_config(cfg)
    cc = cont.ContinuationControl.from_config(cfg)
    tol_root = float(cfg["tolerances"]["tol_root"])
    inf_rel = float(cfg["tolerances"]["inference_rel"])
    run_all = bool(cfg["run_all_starts"])
    order = [float(t) for t in cfg["cancel_solve_order"]]
    neighbor = {float(k): (None if v is None else float(v)) for k, v in cfg["continuation_neighbor"].items()}
    t0 = time.perf_counter()
    if design is None:
        design = cell.design(cfg.get("contrast_basis", "helmert"))
    y = cell.outcome_pairs()
    pilot = gamma_pilot(design, y, ctl)
    merit, merit_source = build_metric(design, pilot)
    admission = None
    if pilot.ok and merit is not None:  # second-stage certification of the same candidate
        admission = admit_with_metric(pilot.certificate, merit, ctl)
    admitted = admission is not None and admission.ok
    pilot_status = pilot.status if (not pilot.ok or admitted or merit is None) else "root_check_failed"
    fit = CellFit(pilot_status, merit_source, time.perf_counter() - t0, {}, {},
                  n_obs=design.n_obs, n_pairs=design.G, d=design.d, inference_rel=inf_rel,
                  pilot=_attempt("zero", pilot, phase="pilot", target_tau=2.0, solved_tau=2.0),
                  start_policy=start_policy_version(cfg), continuation=cc.describe())
    fit.metric = merit.describe() if merit is not None else {"metric_source": merit_source}
    if admission is not None:
        fit.pilot_admission = {"ok": bool(admission.ok), "metric_scope": admission.metric_scope,
                               "metric_residual": float(admission.metric_residual),
                               "metric_step": float(admission.metric_step),
                               "resid_scaled": float(admission.resid_scaled),
                               "newton_corr_norm": float(admission.newton_corr_norm),
                               "rcond": float(admission.rcond), "failed_gates": list(admission.failed_gates)}
    if merit is None or not pilot.ok or not admitted:
        # Without the (admitted) Gamma root there is no admissible first start,
        # and later starts depend on earlier accepted roots: everything is not_attempted.
        why = (f"gamma pilot {pilot.status}" if not pilot.ok else
               f"gamma pilot not admitted under the fixed metric ({admission.message})" if admission is not None
               else f"gamma pilot {pilot.status}")
        msg = f"no admissible start: {why}; merit {merit_source}"
        for tau in order:
            for kind in ("C", "J"):
                if kind == "J" and tau == 1.0:
                    continue
                fit.roots[(kind, tau)] = RootRecord(
                    tau, kind, message=msg,
                    continuation_status=cont.NOT_APPLICABLE if kind == "C" else "no_admissible_start")
                fit.post[(kind, tau)] = PostRecord(status_inference="skipped_root_failed",
                                                   status_correction="skipped_root_failed")
        return fit

    # ---- cancelled roots --------------------------------------------------
    for tau in order:
        rec = RootRecord(tau, "C", continuation_status=cont.NOT_APPLICABLE)
        starts = []
        if pilot.ok:
            starts.append(("gamma_pilot", pilot.theta))
        nb = neighbor.get(tau)
        if nb is not None and fit.roots.get(("C", nb)) is not None and fit.roots[("C", nb)].status == "ok":
            starts.append((f"neighbor_tau{nb:g}", fit.roots[("C", nb)].theta))
        if tau != 1.0 and fit.roots.get(("C", 1.0)) is not None and fit.roots[("C", 1.0)].status == "ok":
            starts.append(("ppml_root", fit.roots[("C", 1.0)].theta))
        ev_fun = _cancel_eval(design, y, tau)
        executed = []  # (label, start vector) of real Newton calls, in priority order
        solved = []  # SolveResult per executed start, in priority order
        stopped = False
        for label, th in starts:  # records are appended only as each start is processed
            if stopped:
                rec.alternatives.append(Attempt(
                    label, "not_attempted", disposition="not_attempted", phase="cancelled", target_tau=tau,
                    solved_tau=tau, message="an earlier start was accepted and run_all_starts is false"))
                continue
            dup = next((lab for lab, t2 in executed if np.array_equal(t2, th)), None)
            if dup is not None:  # identical vector of an EXECUTED start: a reference, not a solve
                rec.alternatives.append(Attempt(label, f"reused:{dup}", disposition="reused", phase="cancelled",
                                                target_tau=tau, solved_tau=tau))
                continue
            res = newton_solve(ev_fun, th, ctl, merit=merit, start_label=label)
            executed.append((label, th))
            solved.append(res)
            rec.attempts += 1
            rec.attempts_ok += int(res.ok)
            rec.newton_calls += 1
            rec.alternatives.append(_attempt(label, res, phase="cancelled", target_tau=tau, solved_tau=tau))
            if res.ok and not run_all:
                stopped = True
        accepted = next((r for r in solved if r.ok), None)
        # reported attempt: first accepted, else the first-priority failure
        reported = accepted if accepted is not None else (solved[0] if solved else None)
        if reported is not None:
            # root time = the reported attempt; every other attempt is retry time
            rec.elapsed_root_s = reported.elapsed_s
            rec.elapsed_retry_s = float(sum(r.elapsed_s for r in solved if r is not reported))
            rec.iterations, rec.backtracks = reported.iterations, reported.backtracks
            rec.terminal, rec.start_used = reported.terminal, reported.start_label
            rec.certified_full_step = bool(reported.certified_full_steps)
            rec.metric_scope = reported.metric_scope
        if accepted is not None:
            rec.status, rec.theta, rec.cert = "ok", accepted.theta, accepted.certificate
            oks = [a.beta for a in rec.alternatives if a.status == "ok"]
            diff = max(abs(b - float(accepted.theta[0])) for b in oks)
            rec.ambiguity_max_diff = diff
            rec.ambiguous = bool(diff > tol_root)
            st = accepted.certificate.evaluation.state
            fit.w_range[tau] = (float(st.w.min()), float(st.w.max()))
        elif reported is not None:
            rec.status, rec.message = reported.status, f"{reported.start_label}: {reported.message}"
        else:
            rec.status, rec.message = "not_attempted", "no admissible start"
        fit.roots[("C", tau)] = rec

    # ---- joint roots (tau != 1), pass 1: the prescribed direct attempt ---------
    jev = lambda tau: _joint_eval(design, y, tau)  # noqa: E731
    for tau in order:
        if tau == 1.0:
            continue
        rec = RootRecord(tau, "J")
        c = fit.roots[("C", tau)]
        if c.status == "ok":
            label, th = f"cancelled_tau{tau:g}", c.theta
        elif pilot.ok:
            label, th = "gamma_pilot", pilot.theta
        else:
            fit.roots[("J", tau)] = rec
            rec.message, rec.continuation_status = "no admissible start", "no_admissible_start"
            continue
        res = newton_solve(jev(tau), th, ctl, merit=merit, start_label=label)
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
        fit.roots[("J", tau)] = rec

    # ---- joint roots, pass 2: bounded continuation for FAILED direct targets ----
    if cc.enabled:
        # frozen seed snapshot: certified direct J roots + C(1) = J(1); never the
        # Gamma pilot; later continuation results are never added
        pool = {}
        for tau in order:
            r = fit.roots.get(("J", tau))
            if r is not None and r.status == "ok":
                pool[float(tau)] = (np.array(r.theta, copy=True), f"J{tau:g}_direct")
        c1 = fit.roots.get(("C", 1.0))
        if c1 is not None and c1.status == "ok":
            pool[1.0] = (np.array(c1.theta, copy=True), "C1_identity_J1")
        for tau in order:
            rec = fit.roots.get(("J", tau))
            if tau == 1.0 or rec is None or rec.status == "ok" or rec.continuation_status:
                continue
            _continue_joint(rec, tau, pool, cc, lambda t, th0, lab: newton_solve(
                jev(t), th0, ctl, merit=merit, start_label=lab))

    # ---- post-processing ----------------------------------------------------
    for key, rec in fit.roots.items():
        post = PostRecord()
        fit.post[key] = post
        if rec.status != "ok":
            post.status_inference = post.status_correction = "skipped_root_failed"
            continue
        ev, fac = rec.cert.evaluation, rec.cert.factor
        t1 = time.perf_counter()
        try:
            inf = pair_sandwich(design, ev, fac, inf_rel, factor=ev.factor if key[0] == "J" else None)
            post.se, post.status_inference = inf.se, inf.status
            post.resid_target_over_se = inf.resid_target_over_se
            post.target_resid = inf.target_resid
            post.schur_beta, post.schur_beta_over_xx = inf.schur_beta, inf.schur_beta_over_xx
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            post.status_inference, post.message = "failed", str(exc)
            inf = None
        post.elapsed_cov_s = time.perf_counter() - t1
        if key[0] == "C":
            t2 = time.perf_counter()
            try:
                adj = inf.adj if inf is not None else fac.solve_t(target_row(design.d))
                cor = manuscript_correction(design, ev, fac, adj)
                post.bias, post.term1, post.term2 = cor.bias, cor.term1, cor.term2
                post.status_correction = "ok"
                # keep the d x d inverse only where the WZ adapter reuses it
                post.Hinv = cor.Hinv if key == ("C", 1.0) else None
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                post.status_correction, post.message = "failed", str(exc)
            post.elapsed_corr_s = time.perf_counter() - t2
        else:
            post.status_correction = "not_attempted"

    # ---- PPML-only extras -----------------------------------------------------
    c1 = fit.roots.get(("C", 1.0))
    wz_cfg = cfg.get("wz", {})
    if c1 is not None and c1.status == "ok" and wz_cfg.get("enabled", True):
        t3 = time.perf_counter()
        try:
            Hinv = fit.post[("C", 1.0)].Hinv
            if Hinv is None:
                Hinv = c1.cert.factor.inverse()
            fit.wz = wz_correction(design, y, c1.cert.evaluation, Hinv,
                                   n_countries=max(design.No, design.Nd))
            fit.wz_status = "ok" if fit.wz.status_bias == "ok" else "failed"
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            fit.wz_status = "failed"
            fit.wz = None
            fit.post[("C", 1.0)].message += f" wz:{exc}"
        fit.elapsed_wz_s = time.perf_counter() - t3
    if c1 is not None and c1.status == "ok" and wz_cfg.get("spj", True):
        t4 = time.perf_counter()
        fit.spj_metric_scope = METRIC_SCOPE_PPML_SUBFIT
        betas, ok = [], True
        for name, op, dp in spj_quarters(cell):
            sub = cell.subset(op, dp)
            sdes = sub.design(cfg.get("contrast_basis", "helmert"))
            res = ppml_from_zero(sdes, sub.outcome_pairs(), ctl)
            fit.spj_quarters.append((name, res.status, float(res.theta[0]) if res.ok else np.nan, res.iterations))
            fit.spj_results.append((name, res))
            if res.ok:
                betas.append(float(res.theta[0]))
            else:
                ok = False
        fit.elapsed_spj_s = time.perf_counter() - t4
        if ok:
            fit.spj_beta = spj_combine(float(c1.theta[0]), betas)
            fit.spj_status = "ok"
        else:
            fit.spj_status = "failed"
    return fit


def _accept_joint(rec, res):
    rec.theta, rec.cert = res.theta, res.certificate
    Hj = res.certificate.evaluation.H
    rec.jac_asym = float(np.max(np.abs(Hj - Hj.T)) / np.max(np.abs(Hj)))
    if not rec.jac_asym < JOINT_SYMMETRY_TOL:  # mandatory diagnostic (contract 4)
        rec.message = f"joint Jacobian asymmetry {rec.jac_asym:.2e} >= {JOINT_SYMMETRY_TOL:g}"


def _continue_joint(rec, tau, pool, cc, solve):
    """Pass 2 for one failed J target; mutates ``rec`` (direct attempt kept)."""
    seed = cont.select_seed(tau, pool)
    if seed is None:
        rec.continuation_status = cont.NO_SEED
        rec.message = f"{rec.start_used}: {rec.message}; continuation: no certified seed"[:300]
        return
    out = cont.continue_to_target(solve, tau, seed, cc)
    rec.continuation_seed, rec.continuation_status = out.seed_label, out.status
    rec.continuation_last_tau = out.last_certified_tau
    for call in out.calls:
        r = call.result
        rec.alternatives.append(_attempt(call.start_label, r, phase=call.phase, target_tau=call.target_tau,
                                         solved_tau=call.solved_tau, source_tau=call.source_tau,
                                         anchor=call.anchor, trial_step=call.trial_step,
                                         step_reductions=call.step_reductions))
        rec.newton_calls += 1
        if call.phase == "endpoint":  # a call of the requested target equation
            rec.attempts += 1
            rec.attempts_ok += int(r.ok)
        else:
            rec.attempts_intermediate += 1
            rec.attempts_intermediate_ok += int(r.ok)
    direct_elapsed = rec.elapsed_root_s
    path_elapsed = float(sum(c.result.elapsed_s for c in out.calls))
    if out.status == cont.ENDPOINT_OK:
        res = out.endpoint
        rec.status, rec.iterations, rec.backtracks = "ok", res.iterations, res.backtracks
        rec.start_used, rec.terminal, rec.metric_scope = "joint_continuation", res.terminal, res.metric_scope
        rec.certified_full_step = bool(res.certified_full_steps)
        rec.elapsed_root_s = res.elapsed_s  # the accepted endpoint call
        rec.elapsed_retry_s = float(direct_elapsed + path_elapsed - res.elapsed_s)
        rec.message = ""
        _accept_joint(rec, res)
        rec.ambiguity_max_diff = np.nan  # no comparison made: branch selection unverified
        note = (f"joint_continuation from {out.seed_label}: {len(out.calls)} call(s); "
                f"branch selection unverified")
        rec.message = (note + ("; " + rec.message if rec.message else ""))[:300]
    else:  # the direct failure stays the reported attempt
        rec.elapsed_retry_s = float(rec.elapsed_retry_s + path_elapsed)
        rec.message = f"{rec.start_used}: {rec.message}; {out.message}"[:300]


def failed_cell_fit(cell, cfg, message):
    """A CellFit in which every root failed before solving (e.g. the design or
    the merit metric could not be built).  Rows are still emitted."""
    No, Nd, T = cell.shape
    G = len(cell.pairs()[0])
    fit = CellFit("numerical_failure", "", float("nan"), {}, {}, n_obs=G * T, n_pairs=G,
                  d=1 + (No + Nd - 1) * (T - 1))
    from ..versions import start_policy_version

    fit.start_policy = start_policy_version(cfg)
    fit.continuation = cont.ContinuationControl.from_config(cfg).describe()
    for tau in (float(t) for t in cfg["cancel_solve_order"]):
        for kind in ("C", "J"):
            if kind == "J" and tau == 1.0:
                continue
            fit.roots[(kind, tau)] = RootRecord(tau, kind, status="numerical_failure", message=message[:200],
                                                continuation_status=cont.NOT_APPLICABLE if kind == "C"
                                                else "no_admissible_start")
            fit.post[(kind, tau)] = PostRecord(status_inference="skipped_root_failed",
                                               status_correction="skipped_root_failed")
    return fit


__all__ = ["Attempt", "build_metric", "failed_cell_fit", "fit_cell", "CellFit", "RootRecord", "PostRecord", "SOLVER_VERSION", "gamma_pilot", "ppml_from_zero"]
