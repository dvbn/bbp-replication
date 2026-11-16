"""CellFit -> result rows, failure index and per-draw diagnostics.

Every draw-cell yields the same set of rows whatever fails (failure rows keep
all available fields; an accepted coefficient survives SE/correction failure).
Rows per draw-cell, for every configured tau: C(tau), CB(tau), J(tau != 1), plus
at tau=1 the PPML-only SE versions of C/CB, WZ-bc x3 and WZ-spj x3
(schema.expected_keyset; 30 rows for the seven-power grid).
Shared setup time (design + Gamma pilot) is recorded once, on the C(2) raw
pair_sandwich row; WZ / SPJ time once on their first row.
"""

from __future__ import annotations

import math

from ..schema import nan_row, validate_row
from ..versions import ACCEPTANCE_RULE_VERSION, SOLVER_VERSION


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return math.nan
    return x


def _root_fields(rec, post, fit):
    """Root-stage fields.  resid_scaled / resid_merit / newton_corr_norm / rcond
    are the contract's terminal diagnostics: the fresh certificate for an
    accepted root, the last iterate of the reported attempt for a failed one
    (status_root says which; a failed root's values never certify it)."""
    term = rec.terminal
    wmin, wmax = fit.w_range.get(rec.tau, (math.nan, math.nan)) if rec.kind == "C" else (math.nan, math.nan)
    return dict(
        solver_version=SOLVER_VERSION,
        acceptance_rule_version=ACCEPTANCE_RULE_VERSION,
        start_policy_version=fit.start_policy,
        start_used=rec.start_used,
        attempts=int(rec.attempts),
        attempts_ok=int(rec.attempts_ok),
        attempts_intermediate=int(rec.attempts_intermediate),
        attempts_intermediate_ok=int(rec.attempts_intermediate_ok),
        newton_calls=int(rec.newton_calls),
        continuation_status=rec.continuation_status,
        continuation_seed=rec.continuation_seed,
        continuation_last_tau=_f(rec.continuation_last_tau),
        iterations=int(rec.iterations),
        backtracks=int(rec.backtracks),
        certified_full_step=bool(rec.certified_full_step),
        status_root=rec.status,
        resid_scaled=_f(term.resid_scaled),
        resid_merit=_f(term.resid_merit),
        newton_corr_norm=_f(term.newton_corr_norm),
        rcond=_f(term.rcond),
        resid_metric=_f(term.metric_residual),
        newton_corr_metric=_f(term.metric_step),
        metric_scope=rec.metric_scope,
        metric_normalization=(fit.metric or {}).get("metric_normalization", ""),
        beta_last_iterate=_f(term.beta),
        theta_max_abs_last=_f(term.theta_max_abs),
        schur_beta=_f(post.schur_beta),
        schur_beta_over_xx=_f(post.schur_beta_over_xx),
        w_min=_f(wmin),
        w_max=_f(wmax),
        root_ambiguous=bool(rec.ambiguous),
        ambiguity_max_diff=_f(rec.ambiguity_max_diff),
        jac_asym=_f(rec.jac_asym),
        merit_source=fit.merit_source,
        resid_target_over_se=_f(post.resid_target_over_se),
        elapsed_root_s=_f(rec.elapsed_root_s),
        elapsed_retry_s=_f(rec.elapsed_retry_s),
    )


def build_rows(fit, base):
    """base: dict with packet, archive_version, dgp, N, T, rep, offdiag,
    draw_hash, config_hash, code_hash."""
    common = dict(base, n_obs=fit.n_obs, n_pairs=fit.n_pairs, d=fit.d)
    rows, setup_done = [], False
    taus = sorted({k[1] for k in fit.roots if k[0] == "C"})
    wz = fit.wz
    for tau in taus:
        rec, post = fit.roots[("C", tau)], fit.post[("C", tau)]
        rf = _root_fields(rec, post, fit)
        beta = float(rec.theta[0]) if rec.status == "ok" else math.nan
        corr_ok = post.status_correction == "ok"
        shared = dict(
            common, **rf, tau=float(tau), bias_hat=_f(post.bias), bias_term1=_f(post.term1),
            bias_term2=_f(post.term2), status_inference=post.status_inference,
            status_correction=post.status_correction, elapsed_cov_s=_f(post.elapsed_cov_s),
            elapsed_corr_s=_f(post.elapsed_corr_s), note=post.message[:200],
        )
        if tau == 1.0 and wz is not None:
            shared.update(wz_bias=_f(wz.bias), wz_bias_no_factor=_f(wz.bias_no_factor))
        first = dict(shared, estimator_version="C", coef_version="raw", se_version="pair_sandwich",
                     beta=beta, se=_f(post.se))
        if not setup_done and tau == 2.0:
            first["elapsed_setup_s"] = _f(fit.elapsed_setup_s)
            first["note"] = (f"pilot={fit.pilot_status};" + first["note"])[:200]
            setup_done = True
        rows.append(first)
        b_bc = beta - post.bias if (rec.status == "ok" and corr_ok) else math.nan
        rows.append(dict(shared, estimator_version="CB", coef_version="manuscript_bc",
                         se_version="pair_sandwich", beta=b_bc, se=_f(post.se)))
        if tau == 1.0:
            rows.extend(_ppml_rows(fit, shared, beta, b_bc))
    for tau in taus:
        if tau == 1.0:
            continue
        rec, post = fit.roots[("J", tau)], fit.post[("J", tau)]
        rf = _root_fields(rec, post, fit)
        beta = float(rec.theta[0]) if rec.status == "ok" else math.nan
        rows.append(dict(
            common, **rf, tau=float(tau), estimator_version="J", coef_version="raw",
            se_version="joint_naive", beta=beta, se=_f(post.se),
            status_inference=post.status_inference, status_correction="not_attempted",
            elapsed_cov_s=_f(post.elapsed_cov_s), note=(rec.message or post.message)[:200],
        ))
    out = []
    for r in rows:
        _single_count_timing(r)
        full = nan_row(**r)
        validate_row(full)
        out.append(full)
    return out


def _single_count_timing(r):
    """Each timed piece of work is carried by exactly one row: root/retry/cov
    time by the C or J row with its own sandwich, correction time by the CB
    pair_sandwich row (setup / WZ / SPJ carriers are set where built)."""
    est, se_v = r["estimator_version"], r["se_version"]
    if not (est in ("C", "J") and se_v in ("pair_sandwich", "joint_naive")):
        for k in ("elapsed_root_s", "elapsed_retry_s", "elapsed_cov_s"):
            r[k] = math.nan
        for k in ("newton_calls", "attempts_intermediate", "attempts_intermediate_ok"):
            r[k] = -1  # work counts carried once, like the timings
    if not (est == "CB" and se_v == "pair_sandwich"):
        r["elapsed_corr_s"] = math.nan


def _ppml_rows(fit, shared, beta, b_bc):
    """PPML-only SE versions and the two WZ coefficient versions (tau = 1)."""
    wz = fit.wz
    root_ok = math.isfinite(beta)
    se_un = _f(wz.se_unadjusted) if wz is not None else math.nan
    se_adj = _f(wz.se_adjusted) if wz is not None else math.nan
    if not root_ok:
        st_un = st_adj = shared["status_inference"]  # skipped_root_failed
    else:
        st_un = "ok" if math.isfinite(se_un) else ("failed" if wz is not None else fit.wz_status)
        st_adj = wz.status_cov if wz is not None else fit.wz_status
        # same inference-accuracy rule as the pair sandwich: |a'S| <= rel * SE
        resid = _f(fit.post[("C", 1.0)].target_resid)
        if not math.isfinite(resid):
            se_un = se_adj = math.nan
            st_un = st_adj = "failed"
        else:
            if st_un == "ok" and resid > fit.inference_rel * se_un:
                se_un, st_un = math.nan, "inference_check_failed"
            if st_adj == "ok" and resid > fit.inference_rel * se_adj:
                se_adj, st_adj = math.nan, "inference_check_failed"
    rows = []
    for est, coef, b in (("C", "raw", beta), ("CB", "manuscript_bc", b_bc)):
        rows.append(dict(shared, estimator_version=est, coef_version=coef, se_version="wz_unadjusted",
                         beta=b, se=se_un, status_inference=st_un))
        rows.append(dict(shared, estimator_version=est, coef_version=coef, se_version="wz_adjusted",
                         beta=b, se=se_adj, status_inference=st_adj))
    pair = shared
    b_wz = beta - wz.bias if (wz is not None and wz.status_bias == "ok" and root_ok) else math.nan
    if not root_ok:
        wz_corr_status = "skipped_root_failed"
    else:
        wz_corr_status = "ok" if math.isfinite(b_wz) else ("failed" if fit.wz_status != "not_attempted" else "not_attempted")
    base_pair_se = _f(fit.post[("C", 1.0)].se)
    base_pair_status = fit.post[("C", 1.0)].status_inference
    for k, (se_v, se, st) in enumerate((("pair_sandwich", base_pair_se, base_pair_status),
                                        ("wz_unadjusted", se_un, st_un), ("wz_adjusted", se_adj, st_adj))):
        r = dict(pair, estimator_version="WZ-bc", coef_version="wz_bc", se_version=se_v, beta=b_wz, se=se,
                 bias_hat=_f(wz.bias) if wz is not None else math.nan, status_inference=st,
                 status_correction=wz_corr_status, elapsed_corr_s=math.nan, elapsed_cov_s=math.nan)
        if k == 0:
            r["elapsed_extra_s"] = _f(fit.elapsed_wz_s)
        rows.append(r)
    spj_note = ";".join(f"{n}:{s}" for n, s, _, _ in fit.spj_quarters)[:200]
    for k, (se_v, se, st) in enumerate((("pair_sandwich", base_pair_se, base_pair_status),
                                        ("wz_unadjusted", se_un, st_un), ("wz_adjusted", se_adj, st_adj))):
        r = dict(pair, estimator_version="WZ-spj", coef_version="spj", se_version=se_v,
                 beta=_f(fit.spj_beta), se=se, bias_hat=math.nan, status_inference=st,
                 status_correction=fit.spj_status if root_ok else "skipped_root_failed",
                 elapsed_corr_s=math.nan, elapsed_cov_s=math.nan, note=spj_note)
        if k == 0:
            r["elapsed_extra_s"] = _f(fit.elapsed_spj_s)
        rows.append(r)
    return rows


FAILURE_FIELDS = ("packet", "archive_version", "dgp", "N", "T", "rep", "estimator_version", "tau",
                  "coef_version", "se_version", "stage", "status")


def failures_from_rows(rows):
    """Stage-specific failure index.  A root that was not attempted (no
    admissible start, e.g. failed Gamma pilot) is a root-stage failure; a
    not_attempted inference/correction stage (e.g. J has no correction) is not."""
    out = []
    for r in rows:
        for stage, key in (("root", "status_root"), ("inference", "status_inference"),
                           ("correction", "status_correction")):
            st = r[key]
            if st not in ("ok", "") and not (st == "not_attempted" and stage != "root"):
                out.append({**{f: r[f] for f in FAILURE_FIELDS[:10]}, "stage": stage, "status": st})
    return out


def _num(v):
    return None if not math.isfinite(_f(v)) else _f(v)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float)):
        return _num(obj) if isinstance(obj, float) else obj
    return _num(obj)


def _attempt_json(a):
    t = a.terminal
    num = _num
    return {"start": a.label, "status": a.status, "disposition": a.disposition, "phase": a.phase,
            "target_tau": num(a.target_tau), "solved_tau": num(a.solved_tau),
            "source_tau": num(a.source_tau), "anchor": a.anchor, "trial_step": num(a.trial_step),
            "step_reductions": a.step_reductions,
            "beta": num(a.beta), "iterations": a.iterations,
            "backtracks": a.backtracks, "elapsed_s": num(a.elapsed_s), "message": a.message[:200],
            "certified_full_step": a.certified_full_step,
            "acceptance_route": ("certified_full_step" if a.certified_full_step else
                                 "iterative_stop_and_fresh_certificate" if a.status == "ok" else ""),
            "metric_scope": a.metric_scope,
            "certificate": _clean(a.accepted_cert) if a.accepted_cert is not None else None,
            "rejected_full_step": _clean(a.rejected_full_step) if a.rejected_full_step is not None else None,
            "last_iterate": {"beta": num(t.beta), "theta_max_abs": num(t.theta_max_abs),
                             "resid_scaled": num(t.resid_scaled), "resid_merit": num(t.resid_merit),
                             "newton_corr_norm": num(t.newton_corr_norm), "rcond": num(t.rcond),
                             "metric_residual": num(t.metric_residual), "metric_step": num(t.metric_step)}}


def certificate_diagnostics(cert):
    """JSON-safe certificate diagnostics, without cached evaluation/LU objects."""
    if cert is None:
        return None
    return _clean({
        "ok": bool(cert.ok), "status": cert.status, "message": cert.message,
        "resid_scaled": cert.resid_scaled, "resid_merit": cert.resid_merit,
        "newton_corr_norm": cert.newton_corr_norm, "rcond": cert.rcond,
        "metric_residual": cert.metric_residual, "metric_step": cert.metric_step,
        "metric_scope": cert.metric_scope, "failed_gates": list(cert.failed_gates),
        "newton_correction": cert.step.tolist() if cert.step is not None else None,
    })


def solve_result_diagnostics(res):
    """One actual Newton call's complete diagnostic record (not a new solve).

    Includes failed terminal state, trajectory, fresh-certificate diagnostics,
    and rejected full-step diagnostics. Cached equation states/LU factors are
    excluded. A call's elapsed_s is contained in its enclosing setup/SPJ/wall
    timer; callers must not add it again to that timer.
    """
    return _clean({
        "status": res.status, "ok": bool(res.ok), "start": res.start_label,
        "theta": res.theta.tolist(), "beta": float(res.theta[0]) if res.ok else None,
        "iterations": res.iterations, "backtracks": res.backtracks,
        "elapsed_s": res.elapsed_s, "message": res.message, "history": res.history,
        "metric_scope": res.metric_scope, "certified_full_steps": res.certified_full_steps,
        "acceptance_route": ("certified_full_step" if res.certified_full_steps else
                             "initial_stop_and_fresh_certificate" if res.ok and res.iterations == 0 else
                             "iterative_stop_and_fresh_certificate" if res.ok else ""),
        "certificate": certificate_diagnostics(res.certificate),
        "rejected_full_step": res.rejected_full_step,
        "last_iterate": dict(vars(res.terminal)),
    })


def diagnostics(fit, base):
    """JSON-serialisable per-draw-cell diagnostics: every attempt of every root
    (accepted, failed, reused or not attempted, in priority/chronological
    order with last-iterate diagnostics; for J the direct attempt and every
    continuation call), the Gamma pilot and its fixed-metric admission."""
    roots = {}
    for (kind, tau), rec in fit.roots.items():
        roots[f"{kind}{tau:g}"] = {
            "status": rec.status, "start_used": rec.start_used, "attempts": rec.attempts,
            "attempts_ok": rec.attempts_ok, "attempts_intermediate": rec.attempts_intermediate,
            "attempts_intermediate_ok": rec.attempts_intermediate_ok, "newton_calls": rec.newton_calls,
            "continuation_status": rec.continuation_status, "continuation_seed": rec.continuation_seed,
            "continuation_last_tau": _num(rec.continuation_last_tau),
            "elapsed_root_s": _num(rec.elapsed_root_s), "elapsed_retry_s": _num(rec.elapsed_retry_s),
            "attempts_detail": [_attempt_json(a) for a in rec.alternatives],
            "ambiguous": rec.ambiguous, "message": rec.message[:300],
        }
    return {
        **{k: base[k] for k in ("packet", "archive_version", "dgp", "N", "T", "rep", "draw_hash")},
        "pilot_status": fit.pilot_status, "merit_source": fit.merit_source,
        "pilot": _attempt_json(fit.pilot) if fit.pilot is not None else None,
        "pilot_admission": _clean(fit.pilot_admission), "metric": _clean(fit.metric),
        "acceptance_rule_version": ACCEPTANCE_RULE_VERSION, "solver_version": SOLVER_VERSION,
        "start_policy_version": fit.start_policy, "joint_continuation": _clean(fit.continuation),
        "spj_metric_scope": fit.spj_metric_scope,
        "wz_status": fit.wz_status, "spj_status": fit.spj_status,
        "spj_quarters": [[n, s, (None if not math.isfinite(b) else b), it] for n, s, b, it in fit.spj_quarters],
        "spj_quarters_detail": [{"quarter": name, **solve_result_diagnostics(res)}
                                for name, res in fit.spj_results],
        "spj_newton_calls": len(fit.spj_results),
        "spj_solve_s": _num(sum(res.elapsed_s for _, res in fit.spj_results)),
        "elapsed_spj_s": _num(fit.elapsed_spj_s),
        "spj_timing_convention": "spj_solve_s is contained in elapsed_spj_s; do not add both",
        "wz_min_leverage_singular": (fit.wz.min_leverage_singular if fit.wz is not None else None),
        "roots": roots,
    }
