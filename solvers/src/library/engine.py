"""Direct-only BBP research fits, with independently audited termination.

The established SciPy/sklearn numerical loops are retained. Newton and Fisher
do not silently switch to L-BFGS. The common audit is outside fitting timers.
No approximation, outcome deletion, ridge repair, or alternative start is used.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve

PREVIOUS = Path(__file__).resolve().parent
sys.path.insert(0, str(PREVIOUS))
from experiment import PointLoss
from vendor_linear_loss import LinearModelLoss
from vendor_newton import NewtonCholeskySolver
from vendor_lbfgsb import _minimize_lbfgsb

STATUS = {0: "verified_complete", 1: "premature_native_stop", 2: "iteration_limit",
          3: "evaluation_limit", 4: "line_search_failure", 5: "factorization_failure",
          6: "non_descent_direction", 7: "numerical_breakdown", 8: "fe_inner_limit",
          9: "absorption_limit", 10: "precision_stagnation", 11: "all_zero_outcome"}
STATUS_CODE = {value: key for key, value in STATUS.items()}
STRICT = 1
FLAG = {"strict_accuracy": STRICT, "first_attempt_native_success": 2, "same_backend_restart": 4,
        "native_warning": 8, "retained_better_first_attempt": 16,
        "nonfinite_audit": 32, "curvature_rank_loss": 64,
        "common_gate_pass_native_unsuccessful": 128,
        "restart_no_measurable_progress": 1024, "missing_profile": 2048,
        "audit_exception": 4096, "nonfinite_intercept": 8192}


class EvaluationLimit(RuntimeError):
    pass


class CountedPointLoss(PointLoss):
    """Count loss/derivative evaluations and enforce the declared work limit."""
    def __init__(self, tau, y, fisher, maxfun):
        super().__init__(tau, y, fisher=fisher)
        self.maxfun = maxfun

    def _budget(self):
        if sum(self.calls.values()) >= self.maxfun:
            raise EvaluationLimit("Pointwise loss/derivative evaluation limit")

    def loss(self, *args, **kwargs):
        self._budget()
        return super().loss(*args, **kwargs)

    def loss_gradient(self, *args, **kwargs):
        self._budget()
        return super().loss_gradient(*args, **kwargs)

    def gradient_hessian(self, *args, **kwargs):
        self._budget()
        return super().gradient_hessian(*args, **kwargs)


class Workspace:
    def __init__(self, x, y, tau, backend, groups, maxfun):
        self.x, self.y = x, y
        self.point = CountedPointLoss(tau, y, backend == "fisher", maxfun)
        if groups is None:
            self.linear = LinearModelLoss(self.point, fit_intercept=False)
        else:
            from fe_direct import ProfileLinearLoss
            self.linear = ProfileLinearLoss(self.point, x, y, groups)
        self.cached_beta = self.cached_fg = None
        self.accepted_beta = None
        self.accepted_iterations = 0
        self.accepted_profile = None

    def fg(self, beta):
        if self.cached_beta is None or not np.array_equal(beta, self.cached_beta):
            self.cached_fg = self.linear.loss_gradient(beta, self.x, self.y, n_threads=1)
            self.cached_beta = beta.copy()
            if (hasattr(self.linear, "_snapshot") and self.accepted_beta is not None
                    and np.array_equal(beta, self.accepted_beta)):
                self.accepted_profile = self.linear._snapshot()
        return self.cached_fg

    def fun(self, beta):
        return self.fg(beta)[0]

    def jac(self, beta):
        return self.fg(beta)[1]

    def accept(self, intermediate_result):
        self.accepted_beta = intermediate_result.x.copy()
        self.accepted_iterations += 1
        if hasattr(self.linear, "_snapshot"):
            self.accepted_profile = self.linear._snapshot()


class DirectNewton(NewtonCholeskySolver):
    """sklearn loop without a cross-family fallback or rejected final iterate."""
    def __init__(self, **kwargs):
        self.fallbacks = 0
        self.failure_reason = None
        self.accepted_iterations = 0
        super().__init__(**kwargs)

    def inner_solve(self, X, y, sample_weight):
        super().inner_solve(X, y, sample_weight)
        if self.use_fallback_lbfgs_solve:
            self.failure_reason = ("non_descent_direction" if
                getattr(self, "gradient_times_newton", 0.) > 0 else "factorization_failure")

    def line_search(self, X, y, sample_weight):
        # Upstream assigns trial coefficients into self.coef; on a failed search
        # retain the last accepted state, not its final rejected candidate.
        old_coef, old_raw = self.coef.copy(), self.raw_prediction.copy()
        old_loss, old_grad = self.loss_value, self.gradient.copy()
        profile_state = (self.linear_loss._snapshot() if
                         hasattr(self.linear_loss, "_snapshot") else None)
        try:
            super().line_search(X, y, sample_weight)
        except Exception:
            self.coef, self.raw_prediction = old_coef, old_raw
            self.loss_value, self.gradient = old_loss, old_grad
            if profile_state is not None:
                self.linear_loss.fe, self.linear_loss.last_beta, self.linear_loss.last_eta = profile_state
            raise
        if self.use_fallback_lbfgs_solve:
            self.coef, self.raw_prediction = old_coef, old_raw
            self.loss_value, self.gradient = old_loss, old_grad
            if profile_state is not None:
                self.linear_loss.fe, self.linear_loss.last_beta, self.linear_loss.last_eta = profile_state
            self.failure_reason = "line_search_failure"
        else:
            self.accepted_iterations += 1

    def fallback_lbfgs_solve(self, X, y, sample_weight):
        self.fallbacks += 1  # Trigger count, never an actual fallback fit.
        self.converged = False


def audit_plain(x, y, beta, tau):
    """Observed, exact Hessian correction; independent of solver curvature."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eta = x @ beta
        a = np.zeros_like(y)
        pos = y > 0
        a[pos] = np.exp(np.log(y[pos]) + (1-tau)*eta[pos])
        b = np.exp((2-tau)*eta)
        residual = b-a
        gradient = x.T @ residual / len(y)
        weights = (tau-1)*a + (2-tau)*b
        hessian = (x.T*weights) @ x / len(y)
    output = dict(score=float(np.max(np.abs(gradient))), index_step=float("inf"))
    if not (np.isfinite(gradient).all() and np.isfinite(hessian).all()):
        output["audit_flag"] = "nonfinite_audit"
        return output
    try:
        correction = cho_solve(cho_factor(hessian), gradient)
        output["index_step"] = float(np.sqrt(np.mean((x @ correction)**2)))
    except np.linalg.LinAlgError:
        output["audit_flag"] = "curvature_rank_loss"
    return output


def _audit(x, y, beta, eta, tau, groups):
    try:
        if groups is None:
            result = audit_plain(x, y, beta, tau)
        elif eta is None:
            result = dict(score=float("inf"), index_step=float("inf"),
                          audit_flag="missing_profile")
        else:
            from fe_direct import audit_absorbed
            result = audit_absorbed(x, y, eta, tau, groups)
            if result.get("curvature_rank_loss"):
                result["audit_flag"] = "curvature_rank_loss"
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result = dict(score=float("inf"), index_step=float("inf"),
                      audit_flag="audit_exception", audit_error=repr(exc))
    maximum = max(result["score"], result["index_step"])
    result["accepted"] = bool(np.isfinite(maximum) and maximum <= 1e-6)
    result["accepted_strict"] = bool(np.isfinite(maximum) and maximum <= 1e-8)
    return result


def _native_reason(message, success, iterations, maxiter):
    if "EVALUATION" in message or "EVALUATIONS" in message:
        return "evaluation_limit"
    if "ITERATIONS REACHED" in message or iterations >= maxiter:
        return "iteration_limit"
    if "ABNORMAL" in message:
        return "line_search_failure"
    if success:
        return "native_stop"
    return "numerical_breakdown"


def solve(x, y, tau, backend, groups=None, maxiter=20000, maxfun=200000):
    """Complete one exact score using at most two same-backend attempts.

    Restart only a native-success point missing the independent accuracy gate,
    after the first attempt has reduced the initial score measurably. A restart
    that cannot improve the common error is reported as precision stagnation.
    Iteration and evaluation allowances apply to the whole fit, not each attempt.
    """
    if backend not in ("newton", "fisher", "lbfgs"):
        raise ValueError("Unknown backend")
    if not (1 <= tau <= 2):
        raise ValueError("This experiment uses 1 <= tau <= 2")
    x, y = np.asarray(x, float), np.asarray(y, float)
    beta = np.zeros(x.shape[1])
    if y.mean() <= 0:
        return dict(beta=beta, index=None, score=float("inf"), index_step=float("inf"),
                    status="all_zero_outcome", native_success=False, iterations=0,
                    evaluations=0, attempts=0, fallbacks=0, wall_seconds=0., cpu_seconds=0.,
                    flags=FLAG["nonfinite_intercept"], flag_names=["nonfinite_intercept"],
                    status_code=STATUS_CODE["all_zero_outcome"], accepted=False, accepted_strict=False,
                    native_message="No finite intercept for all-zero outcomes")
    if groups is None:
        beta[0] = np.log(y.mean())
    initial_eta = np.full(len(y), np.log(y.mean()))
    initial_residual = np.exp((2-tau)*initial_eta)-y*np.exp((1-tau)*initial_eta)
    initial_score = max(float(np.max(np.abs(x.T @ initial_residual/len(y)))), 1e-300)
    flags, records = [], []
    total_wall = total_cpu = 0.
    total_iterations = fallbacks = 0
    native_success = False
    last_eta = None
    # Construction belongs in the fitting time, including absorbed workspace.
    wall, cpu = time.perf_counter(), time.process_time()
    work = Workspace(x, y, tau, backend, groups, maxfun)
    total_wall += time.perf_counter()-wall
    total_cpu += time.process_time()-cpu
    audit = dict(score=float("inf"), index_step=float("inf"), accepted=False,
                 accepted_strict=False)
    for attempt in range(2):
        remaining = maxiter-total_iterations
        if remaining <= 0:
            break
        start_beta, previous_audit, previous_eta = beta.copy(), audit.copy(), last_eta
        work.cached_beta = work.cached_fg = None
        work.accepted_beta = beta.copy()
        work.accepted_iterations = 0
        work.accepted_profile = (work.linear._snapshot() if groups is not None else None)
        solver = result = None
        reason, message, iterations, native_success = "numerical_breakdown", "", 0, False
        before_evaluations = sum(work.point.calls.values())
        wall, cpu = time.perf_counter(), time.process_time()
        with warnings.catch_warnings(record=True) as caught, np.errstate(over="ignore", invalid="ignore"):
            warnings.simplefilter("always")
            try:
                if backend == "lbfgs":
                    result = _minimize_lbfgsb(work.fun, beta, jac=work.jac,
                        gtol=1e-10, ftol=0., maxls=100, maxiter=remaining,
                        maxfun=max(1, maxfun-before_evaluations), callback=work.accept)
                    beta = result.x.copy()
                    iterations, native_success = int(result.nit), bool(result.success)
                    message = str(result.message)
                    reason = _native_reason(message, native_success, iterations, remaining)
                else:
                    solver = DirectNewton(coef=beta.copy(), linear_loss=work.linear,
                                          tol=1e-10, max_iter=remaining, n_threads=1)
                    beta = solver.solve(x, y, None).copy()
                    iterations = int(solver.iteration)
                    native_success = bool(solver.converged)
                    fallbacks += solver.fallbacks
                    reason = (solver.failure_reason or
                              ("native_stop" if native_success else "iteration_limit"))
                    message = reason
                if groups is not None:
                    last_eta = work.linear.profile(beta).copy()
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                beta = solver.coef.copy() if solver is not None else work.accepted_beta.copy()
                if solver is None and groups is not None and work.accepted_profile is not None:
                    work.linear.fe, work.linear.last_beta, work.linear.last_eta = work.accepted_profile
                iterations = (solver.accepted_iterations if solver is not None else
                              work.accepted_iterations)
                native_success = False
                message = repr(exc)
                if isinstance(exc, EvaluationLimit):
                    reason = "evaluation_limit"
                elif type(exc).__name__ == "FEInnerLimit":
                    reason = ("absorption_limit" if getattr(exc, "stage", None) == "absorption"
                              else "fe_inner_limit")
                elif "absorp" in str(exc).lower() and "converg" in str(exc).lower():
                    reason = "absorption_limit"
                elif "profil" in str(exc).lower() and "converg" in str(exc).lower():
                    reason = "fe_inner_limit"
                else:
                    reason = "numerical_breakdown"
                # Preserve a final index when already cached at the returned
                # coefficient; do not grant uncharged extra solving after a cap.
                if groups is not None:
                    lb = getattr(work.linear, "last_beta", None)
                    last_eta = (work.linear.last_eta.copy() if lb is not None and
                                np.array_equal(beta, lb) else None)
        elapsed_wall, elapsed_cpu = time.perf_counter()-wall, time.process_time()-cpu
        total_wall += elapsed_wall
        total_cpu += elapsed_cpu
        total_iterations += iterations
        audit = _audit(x, y, beta, last_eta, tau, groups)
        record = dict(reason=reason, native_success=native_success, iterations=iterations,
                      evaluations=sum(work.point.calls.values())-before_evaluations,
                      score=audit["score"], index_step=audit["index_step"],
                      wall_seconds=elapsed_wall, cpu_seconds=elapsed_cpu)
        records.append(record)
        if caught:
            flags.append("native_warning")
        if audit.get("audit_flag"):
            flags.append(audit["audit_flag"])
        if audit["accepted"]:
            status = "verified_complete"
            if not native_success:
                flags.append("common_gate_pass_native_unsuccessful")
            break
        error = max(audit["score"], audit["index_step"])
        if attempt:
            old_error = max(previous_audit["score"], previous_audit["index_step"])
            if not np.isfinite(error) or error >= .99*old_error:
                flags.append("restart_no_measurable_progress")
                if old_error < error or not np.isfinite(error):
                    beta, audit, last_eta = start_beta, previous_audit, previous_eta
                    flags.append("retained_better_first_attempt")
            status = "precision_stagnation" if native_success else reason
            break
        if (native_success and np.isfinite(error) and audit["score"] < .99*initial_score
                and total_iterations < maxiter and sum(work.point.calls.values()) < maxfun):
            flags.append("same_backend_restart")
            continue
        status = "premature_native_stop" if native_success else reason
        break
    else:
        status = "precision_stagnation"
    if not records:
        status, message = "iteration_limit", "No outer iteration allowance"
    profile_info = {}
    if groups is not None:
        profile_info = dict(profile_sweeps=int(getattr(work.linear, "sweeps", 0)),
                            absorption_sweeps=int(getattr(work.linear, "absorption_sweeps", 0)))
    if records and records[0]["native_success"]:
        flags.append("first_attempt_native_success")
    if audit["accepted_strict"]:
        flags.append("strict_accuracy")
    return dict(beta=beta, index=last_eta, score=audit["score"], index_step=audit["index_step"],
                status=status, status_code=STATUS_CODE[status], native_success=native_success, iterations=total_iterations,
                evaluations=sum(work.point.calls.values()), attempts=len(records), fallbacks=fallbacks,
                wall_seconds=total_wall, cpu_seconds=total_cpu,
                flags=sum(FLAG.get(flag, 0) for flag in set(flags)), flag_names=sorted(set(flags)),
                accepted=audit["accepted"], accepted_strict=audit["accepted_strict"],
                native_message=message, attempt_records=records, calls=work.point.calls,
                **profile_info)
