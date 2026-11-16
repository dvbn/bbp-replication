"""GPML on Y**omega and paired power-mean slopes, with unchanged WZ roots.

The fitted log-mean slope for Y**omega is gamma; the power-mean slope is
beta = gamma / omega.  Row influences are therefore divided by omega and
covariances by omega**2.  Both outcomes use precisely the same observed rows,
within-country unweighted centering, column scales and two-way CGM convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.stats import norm

from .cohort import build_panel
from .fits import METRIC_ONLY_GATES, _admission, _application_fallback, fit_exponential, joint_country_effects
from .inference import Z95, eig_summary, exp_row_scores, summarize, two_way
from wz_fe_sim.estimators.factorial import RootRecord, _attempt, build_metric
from wz_fe_sim.estimators.newton import METRIC_SCOPE_PILOT_PRELIM, SolverControl, admit_with_metric, factorize, newton_solve


@dataclass
class PowerFit:
    omega: float
    kind: str
    root: object
    gamma: np.ndarray
    beta: np.ndarray
    phi_power: np.ndarray | None
    step_power: np.ndarray
    covariance_gamma: np.ndarray
    covariance_power: np.ndarray
    se_gamma: np.ndarray
    se_power: np.ndarray
    inference_status: np.ndarray
    audit: dict
    fitted_mean: np.ndarray | None = None


def _fit_tau2(panel, cfg):
    """WZ's existing Gamma pilot/admission and C2 -> J2 calls, without other powers.

    This is orchestration only. All Newton steps, certificates, metric creation,
    metric admission and bounded metric refinement are the vendored functions.
    There is no off-power continuation seed in this tau2-only run.
    """
    ctl = SolverControl.from_config(cfg)
    pilot = newton_solve(panel.evaluator("C", 2.0), np.zeros(panel.d), ctl, objective=panel.objective(2),
                         start_label="zero", metric_scope=METRIC_SCOPE_PILOT_PRELIM)
    preliminary = _attempt("zero", pilot, phase="pilot", target_tau=2.0, solved_tau=2.0)
    merit, source = build_metric(panel, pilot)
    admission = admit_with_metric(pilot.certificate, merit, ctl) if pilot.ok and merit is not None else None
    initial, refinement = _admission(admission), None
    if admission is not None and not admission.ok and admission.failed_gates \
            and set(admission.failed_gates) <= METRIC_ONLY_GATES:
        pilot = newton_solve(panel.evaluator("C", 2.0), pilot.theta, ctl, merit=merit,
                             start_label="gamma_metric_refinement")
        refinement = _attempt("gamma_metric_refinement", pilot, phase="pilot", target_tau=2.0, solved_tau=2.0)
        admission = admit_with_metric(pilot.certificate, merit, ctl) if pilot.ok else None
    admitted = bool(merit is not None and pilot.ok and admission is not None and admission.ok)
    out = SimpleNamespace(roots={}, pilot=preliminary, admission_initial=initial, refinement=refinement,
                          admission_final=_admission(admission), pilot_admitted=admitted,
                          metric=merit.describe() if merit is not None else {}, merit_source=source)
    for kind in ("C", "J"):
        rec = RootRecord(2.0, kind, continuation_status="not_applicable_tau2_only")
        out.roots[(kind, 2.0)] = rec
        if not admitted:
            rec.message = "No admissible Gamma pilot under unchanged WZ metric/certificate controls."
            continue
        cancelled = out.roots[("C", 2.0)]
        label, theta = ("cancelled_tau2", cancelled.theta) if kind == "J" and cancelled.status == "ok" \
            else ("gamma_pilot", pilot.theta)
        result = newton_solve(panel.evaluator(kind, 2.0), theta, ctl, merit=merit, start_label=label)
        rec.status, rec.start_used = result.status, label
        rec.attempts, rec.attempts_ok, rec.newton_calls = 1, int(result.ok), 1
        rec.iterations, rec.backtracks, rec.message = result.iterations, result.backtracks, result.message
        rec.terminal, rec.metric_scope = result.terminal, result.metric_scope
        rec.certified_full_step = bool(result.certified_full_steps)
        rec.alternatives.append(_attempt(label, result, phase="cancelled" if kind == "C" else "direct",
                                        target_tau=2.0, solved_tau=2.0))
        if result.ok:
            rec.theta, rec.cert = result.theta, result.certificate
        else:
            _application_fallback(rec, panel.evaluator(kind, 2.0), [("application_zero", np.zeros(panel.d))],
                                  ctl, merit)
    return out


def fit_omega(sample, B, scale, rows, cfg, omega, tau=2.0):
    """Fit C/J at tau 1 or 2 with the historical equations and inference.

    The tau2-only path is unchanged. The tau1 path retains the original Gamma
    pilot, C2 and C1 start sequence; later unused off-grid roots are omitted.
    Joint and cancelled PPML share the exact profile identity.
    """
    if tau not in (1.0, 2.0):
        raise ValueError("this power-mean helper supports tau=1 or tau=2")
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    y = sample.y ** omega
    if not np.array_equal(y == 0, sample.y == 0):
        raise FloatingPointError("power transformation changed the zero mask")
    if not np.isfinite(y).all():
        raise FloatingPointError("non-finite transformed outcome")
    panel = build_panel(rows, B, y)
    ex = _fit_tau2(panel, cfg) if tau == 2.0 else fit_exponential(panel, cfg)
    p = B.shape[1]
    fits = {}
    for kind in ("C", "J"):
        rec = ex.roots[("C" if tau == 1.0 else kind, tau)]
        missing = lambda: np.full(p, np.nan)
        info = {
            "omega": omega, "tau": tau, "kind": kind, "status": rec.status,
            "start_used": rec.start_used, "attempts": rec.attempts,
            "attempts_ok": rec.attempts_ok, "alternatives": rec.alternatives,
            "terminal": rec.terminal, "message": rec.message,
            "continuation_status": rec.continuation_status,
            "coordinates": "Nested attempt beta and terminal beta refer to scaled gamma, not gamma/omega.",
        }
        fit = PowerFit(omega, kind, rec, missing(), missing(), None, missing(),
                       np.full((p, p), np.nan), np.full((p, p), np.nan),
                       missing(), missing(), np.full(p, "fit_failed", dtype=object), info)
        fits[kind] = fit
        if rec.status != "ok":
            continue
        fit.gamma, fit.beta = rec.theta / scale, rec.theta / scale / omega
        cert = rec.cert
        fit.step_power = cert.step / scale / omega
        info["gamma_original"] = fit.gamma
        info["beta_power"] = fit.beta
        info["certificate"] = {
            "ok": bool(cert.ok), "resid_scaled": cert.resid_scaled,
            "newton_corr_norm": cert.newton_corr_norm, "rcond": cert.rcond,
            "metric_residual": cert.metric_residual, "metric_step": cert.metric_step,
            "metric_scope": cert.metric_scope,
        }
        evaluated = panel.evaluator(kind, tau)(rec.theta, True)
        H = evaluated.H
        factor, step = cert.factor, cert.step
        if tau == 1.0 and kind == "J":
            cancelled = panel.evaluator("C", tau)(rec.theta, True)
            info["identity_score_max_abs_diff"] = float(np.max(np.abs(evaluated.S - cancelled.S)))
            info["identity_jacobian_max_abs_diff"] = float(np.max(np.abs(H - cancelled.H)))
            if not np.array_equal(evaluated.S, cancelled.S) or not np.array_equal(H, cancelled.H):
                raise ArithmeticError("PPML joint/cancelled score or observed Jacobian identity failed")
            factor = factorize(H, SolverControl.from_config(cfg).rcond_min)
            step = factor.solve(evaluated.S)
            fit.step_power = step / scale / omega
            info["start_used"] = "identity:C(1);fresh_joint_Jacobian_factorization"
        ev = np.linalg.eigvals(H)
        info["negative_jacobian"] = {
            "min_real_eigenvalue": float(ev.real.min()),
            "max_abs_imag_eigenvalue": float(np.abs(ev.imag).max()),
            "max_asymmetry": float(np.max(np.abs(H - H.T))),
        }
        if kind == "J":
            fit.fitted_mean = np.empty(sample.n)
            for cohort, state in zip(panel.cohorts, panel.states(rec.theta, tau)):
                fit.fitted_mean[cohort.rows.ravel()] = (state.a[:, None] * state.mu).ravel()
            alpha, log_a = joint_country_effects(panel, rec.theta, tau, sample, scale)
            info["country_alpha_gamma_original_X"] = alpha
            info["country_alpha_power_original_X"] = alpha / omega
            info["country_log_a_centered_X"] = log_a
        try:
            phi_gamma = factor.solve(exp_row_scores(panel, rec.theta, tau, kind).T).T / scale
            tw = two_way(phi_gamma, sample.country, sample.family, sample.inter)
            _, V, se, _, _, st = summarize(fit.gamma, tw.cr0, np.ones(p), "cr0")
            ratio = np.full(p, np.nan)
            ok = st == "ok"
            ratio[ok] = np.abs(step[ok] / scale[ok]) / se[ok]
            status = np.asarray(st, dtype=object)
            status[ok & (ratio > cfg["tolerances"]["inference_rel"])] = "inference_check_failed"
            fit.phi_power = phi_gamma / omega
            fit.covariance_gamma, fit.covariance_power = V, V / omega**2
            fit.se_gamma = np.where(status == "ok", se, np.nan)
            fit.se_power = fit.se_gamma / omega
            fit.inference_status = status
            info.update({"inference_status": status, "step_over_se": ratio,
                         "covariance_gamma_cr0": V, "covariance_power_cr0": V / omega**2,
                         "covariance_power_eigen": eig_summary(V / omega**2),
                         "cluster_counts": {k: g for k, (_, g) in tw.meats.items()}})
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            fit.inference_status[:] = f"inference_failed: {exc}"
            info["inference_status"] = fit.inference_status
    setup = {"omega": omega, "n_obs": len(y), "n_zero_outcomes": int((y == 0).sum()),
             "zero_mask_preserved": True, "requested_solve_order": [2.0],
             "gamma_pilot": {"preliminary": ex.pilot, "admission_initial": ex.admission_initial,
                             "refinement": ex.refinement, "admission_final": ex.admission_final,
                             "admitted": ex.pilot_admitted, "metric": ex.metric,
                             "merit_source": ex.merit_source}}
    if tau == 1.0:
        setup["requested_solve_order"] = ex.solve_order
        setup["reported_tau"] = tau
        setup["unreported_numerical_fits"] = {f"{k}{t:g}": {"status": r.status, "alternatives": r.alternatives}
                                              for (k, t), r in ex.roots.items() if t != 1.0}
        cancelled, joint = fits["C"], fits["J"]
        if cancelled.phi_power is not None and joint.phi_power is not None:
            setup["ppml_identity"] = {
                "max_abs_beta_difference": float(np.max(np.abs(cancelled.beta - joint.beta))),
                "max_abs_influence_difference": float(np.max(np.abs(cancelled.phi_power - joint.phi_power))),
                "max_abs_covariance_difference": float(np.max(np.abs(cancelled.covariance_power - joint.covariance_power))),
                "score_max_abs_difference": joint.audit["identity_score_max_abs_diff"],
                "jacobian_max_abs_difference": joint.audit["identity_jacobian_max_abs_diff"],
            }
    return fits, setup


def paired_difference(half, mean, sample, inference_rel):
    """Paired beta_half-beta_mean, retaining cross-outcome cluster covariance.

    The numerical bound adds absolute Newton corrections from both fits; a
    small signed cancellation of numerical errors cannot pass the check.
    """
    delta = half.beta - mean.beta
    p = len(delta)
    empty = np.full(p, np.nan)
    result = {"difference": delta, "se": empty.copy(), "ci95_lo": empty.copy(),
              "ci95_hi": empty.copy(), "z": empty.copy(), "p_value": empty.copy(),
              "status": np.full(p, "fit_or_inference_unavailable", dtype=object),
              "step_bound_over_se": empty.copy(), "variance": empty.copy(),
              "cross_covariance": np.full((p, p), np.nan),
              "difference_covariance": np.full((p, p), np.nan)}
    if half.phi_power is None or mean.phi_power is None:
        return result
    stacked = np.column_stack([half.phi_power, mean.phi_power])
    V = two_way(stacked, sample.country, sample.family, sample.inter).cr0
    cross = V[:p, p:]
    Vdelta = V[:p, :p] + V[p:, p:] - cross - cross.T
    _, _, se, lo, hi, status = summarize(delta, Vdelta, np.ones(p), "difference_cr0")
    ratio = np.full(p, np.nan)
    ok = status == "ok"
    ratio[ok] = (np.abs(half.step_power[ok]) + np.abs(mean.step_power[ok])) / se[ok]
    status = np.asarray(status, dtype=object)
    status[ok & (ratio > inference_rel)] = "inference_check_failed"
    valid = status == "ok"
    z = np.where(valid, delta / np.where(valid, se, 1.0), np.nan)
    result.update({"se": np.where(valid, se, np.nan), "ci95_lo": np.where(valid, lo, np.nan),
                   "ci95_hi": np.where(valid, hi, np.nan), "z": z,
                   "p_value": 2 * norm.sf(np.abs(z)), "status": status,
                   "step_bound_over_se": ratio, "variance": np.diag(Vdelta),
                   "cross_covariance": cross, "difference_covariance": Vdelta,
                   "stacked_covariance": V})
    return result


def transformed_change(beta, se, contrast=1.0):
    """Monotone endpoint transformation; percentages are 100 * proportional change."""
    b = np.asarray(beta)
    lo, hi = (b - Z95 * se) * contrast, (b + Z95 * se) * contrast
    with np.errstate(over="ignore", invalid="ignore"):
        return np.expm1(b * contrast), np.expm1(np.minimum(lo, hi)), np.expm1(np.maximum(lo, hi))


def joint_moment_compatibility(half, mean):
    """Necessary Jensen check on joint fitted levels; no cancelled level claim."""
    if half.fitted_mean is None or mean.fitted_mean is None:
        return {"status": "unavailable", "reason": "joint fitted levels unavailable"}
    ratio = half.fitted_mean**2 / mean.fitted_mean
    finite = np.isfinite(ratio) & (mean.fitted_mean > 0)
    if not finite.all():
        return {"status": "nonfinite_fitted_moment_ratio", "n_nonfinite": int((~finite).sum())}
    tolerance = 1e-10
    violations = ratio > 1 + tolerance
    return {
        "status": "violations" if violations.any() else "no_observed_row_violations",
        "criterion": "fitted E(sqrt(Y)|X,country)^2 / fitted E(Y|X,country) <= 1",
        "ratio_tolerance": tolerance, "n_observations": len(ratio),
        "n_violations": int(violations.sum()), "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()), "ratio_median": float(np.median(ratio)),
        "violating_sample_row_indices": np.flatnonzero(violations),
        "qualification": "Necessary, not sufficient, moment compatibility on observed rows; no repair applied. "
                         "Cancelled normalization does not recover baseline conditional moment levels.",
    }
