"""Actual-Jacobian Newton with a fixed SPD residual merit and Armijo backtracking.

Step:   p solves (DS) p = -S, i.e. H p = S with H = -DS (actual, nonsymmetric,
        signed; never symmetrised, floored, ridged or pseudo-inverted).
Merit:  phi = 1/2 S' W S with W = H_+^{-1} fixed for the whole solve
        (H_+ = Gamma-pilot curvature, or the documented fallback B'B/n).
        For nonsingular DS, grad(phi)'p = -S'WS = -2 phi < 0, so Armijo
        phi(theta + alpha p) <= (1 - 2 c alpha) phi is a valid descent test even
        when DS is nonsymmetric or indefinite.  This is not a global
        convergence guarantee.
Convex variant (Gamma pilot / PPML subfits only): Armijo on the convex
        objective Q with grad Q = -S.

Root certificate (independent of inference): a FRESH evaluation of the
original equation must give
    resid_scaled = max_k |S_k| / scale_k <= tol_scaled    (scale_k = sum of
                   absolute summands of coordinate k; unit free),
    newton_corr_norm = ||H^{-1} S||_inf <= tol_step       (log-index units),
    rcond_1(equilibrated H) > rcond_min,
and, whenever the solve has a fixed metric (every merit-globalised target
solve; ACCEPTANCE_RULE_VERSION), the mean-scale fixed-metric norms
    metric_residual = sqrt((S/n)' A^{-1} (S/n)) <= tol_metric_residual,
    metric_step     = sqrt(p' A p)              <= tol_metric_step,
with p = H^{-1} S the exact Newton correction, n the fitted observation count
and A = H_Gamma / n for the raw-sum Gamma-pilot curvature or A = B'B/n for the
already-averaged design-Gram fallback.  The same conjunction is applied on
every acceptance route (initial stop, final fresh certificate and the
certificate-after-failed-line-search full step); the metric is the solve's
fixed merit matrix and is never rebuilt at a candidate.  Objective-only solves
(Gamma pilot before its metric exists, PPML SPJ subfits) carry no metric and
use the coordinate gates; their ``metric_scope`` says so explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import LinAlgWarning, cho_factor, lu_factor, lu_solve, solve_triangular
from scipy.linalg.lapack import dgecon

import warnings

from ..versions import ACCEPTANCE_RULE_VERSION, SOLVER_VERSION  # noqa: F401  (re-exported)

# Explicit metric normalisations (certificate_norm_reconciliation.md):
#   raw_sum_over_n: supplied matrix M is a raw-sum curvature (H_Gamma); A = M / n
#   already_mean:   supplied matrix M is already averaged (B'B/n);   A = M
METRIC_NORMALIZATIONS = ("raw_sum_over_n", "already_mean")
METRIC_SCOPE_FIXED = "fixed_metric"
METRIC_SCOPE_OBJECTIVE = "objective_coordinate_only"
METRIC_SCOPE_PILOT_PRELIM = "gamma_pilot_preliminary_coordinate_only"
METRIC_SCOPE_PILOT_ADMISSION = "gamma_pilot_fixed_metric_admission"
METRIC_SCOPE_PPML_SUBFIT = "objective_only_ppml_subfit"


@dataclass(frozen=True)
class SolverControl:
    max_iter: int = 200
    max_backtracks: int = 60
    backtrack_factor: float = 0.5
    armijo: float = 1e-4
    tol_scaled: float = 1e-10
    tol_step: float = 1e-9
    rcond_min: float = 1e-13
    tol_metric_residual: float = 1e-10
    tol_metric_step: float = 1e-9

    @classmethod
    def from_config(cls, cfg):
        s, t = cfg["solver"], cfg["tolerances"]
        return cls(
            int(s["max_iter"]),
            int(s["max_backtracks"]),
            float(s["backtrack_factor"]),
            float(s["armijo"]),
            float(t["tol_scaled"]),
            float(t["tol_step"]),
            float(t["rcond_min"]),
            float(t["tol_metric_residual"]),
            float(t["tol_metric_step"]),
        )


class SingularJacobian(Exception):
    """Exact zero pivot or rcond <= rcond_min; ``rcond`` is the estimate when
    one was computed (NaN otherwise) so failed attempts keep it."""

    def __init__(self, message, rcond=np.nan, factor=None):
        super().__init__(message)
        self.rcond = float(rcond)
        self.factor = factor  # LU of an ill-conditioned (not exactly singular) H; diagnostics only


@dataclass
class Factor:
    """LU of the equilibrated H:  H = Ds Hs Ds,  Hs = P L U."""

    lu: tuple
    ds: np.ndarray
    rcond: float

    def solve(self, b):
        """H^{-1} b  (H = diag(s) Hs diag(s), so x = Hs^{-1}(b/s)/s)."""
        s = self.ds[:, None] if b.ndim == 2 else self.ds
        return lu_solve(self.lu, b / s) / s

    def solve_t(self, b):
        """H^{-T} b."""
        x = lu_solve(self.lu, b / self.ds, trans=1)
        return x / self.ds

    def inverse(self):
        n = len(self.ds)
        return self.solve(np.eye(n))


def factorize(H, rcond_min):
    """Equilibrated dense LU with a LAPACK 1-norm reciprocal condition estimate.

    Equilibration Hs = H / (s s'), s_k = sqrt(max(rowmax_k, colmax_k)), is a
    diagonal congruence, so it keeps the sign structure and does not alter the
    Newton step.  Exact zero pivots or rcond <= rcond_min raise SingularJacobian.
    """
    if not np.isfinite(H).all():
        raise FloatingPointError("non-finite Jacobian")
    A = np.abs(H)
    s = np.sqrt(np.maximum(A.max(axis=0), A.max(axis=1)))
    if np.any(s == 0):
        raise SingularJacobian("zero row/column in Jacobian")
    Hs = H / s[:, None] / s[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        try:
            lu = lu_factor(Hs, check_finite=False)
        except LinAlgWarning as exc:  # exactly singular factor
            raise SingularJacobian(str(exc)) from exc
    if np.any(np.diag(lu[0]) == 0):
        raise SingularJacobian("zero pivot", 0.0)
    anorm = np.abs(Hs).sum(axis=0).max()
    rcond, info = dgecon(lu[0], anorm, norm="1")
    if info != 0 or not np.isfinite(rcond):
        raise SingularJacobian("condition estimate failed")
    if rcond <= rcond_min:
        raise SingularJacobian(f"rcond {rcond:.3e} <= rcond_min", rcond, Factor(lu, s, float(rcond)))
    return Factor(lu, s, float(rcond))


class Merit:
    """Fixed SPD matrix M of one solve: globalisation merit and certificate metric.

    Globalisation (unchanged): phi(S) = 1/2 S' M^{-1} S via the lower Cholesky
    factor L of M.  Certificate metric (explicit normalisation): A = c^2 M with
    c^2 = 1/n for a raw-sum curvature (``raw_sum_over_n``) and c^2 = 1 for an
    already-averaged matrix (``already_mean``); its lower factor is c L, so
        metric_residual(S) = ||(c L)^{-1} (S/n)||_2 = sqrt((S/n)' A^{-1} (S/n)),
        metric_step(p)     = ||(c L)' p||_2         = sqrt(p' A p).
    Only the lower triangle returned by cho_factor is used (its other triangle
    is unspecified).  Within a solve the merit and the metric differ by a fixed
    positive scalar, so the Armijo comparison is unaffected."""

    def __init__(self, H_plus, source, n_obs, normalization):
        if normalization not in METRIC_NORMALIZATIONS:
            raise ValueError(f"metric normalization must be one of {METRIC_NORMALIZATIONS}")
        n = int(n_obs)
        if n != n_obs or n < 1:
            raise ValueError("n_obs must be a positive integer observation count")
        Hs = 0.5 * (H_plus + H_plus.T)  # H_+ is SPD by construction; exact symmetrisation
        self.chol = cho_factor(Hs, lower=True, check_finite=True)
        self.L = np.tril(self.chol[0])  # the meaningful triangle only
        self.source = source
        self.n_obs = n
        self.normalization = normalization
        self.c = 1.0 / np.sqrt(n) if normalization == "raw_sum_over_n" else 1.0

    @classmethod
    def gamma_curvature(cls, H_gamma, n_obs):
        """Raw-sum cancelled tau=2 curvature at the Gamma pilot: A = H_Gamma / n."""
        return cls(H_gamma, "gamma_pilot_curvature", n_obs, "raw_sum_over_n")

    @classmethod
    def design_gram_fallback(cls, gram_over_n, n_obs):
        """Already-averaged design Gram B'B/n (documented fallback): A = B'B/n."""
        return cls(gram_over_n, "fallback_BtB_over_n", n_obs, "already_mean")

    def __call__(self, S):
        z = solve_triangular(self.L, S, lower=True, check_finite=False)
        return 0.5 * float(z @ z)

    def metric_residual(self, S):
        z = solve_triangular(self.L, S, lower=True, check_finite=False)
        return float(np.linalg.norm(z) / (self.c * self.n_obs))

    def metric_step(self, p):
        return float(self.c * np.linalg.norm(self.L.T @ p))

    def describe(self):
        return {"metric_source": self.source, "metric_normalization": self.normalization,
                "metric_n_obs": self.n_obs}


def metric_values(merit, S, p):
    """(metric_residual, metric_step); NaN where not computable (never an exception)."""
    if merit is None:
        return np.nan, np.nan
    out = []
    for fn, v in ((merit.metric_residual, S), (merit.metric_step, p)):
        try:
            with np.errstate(all="ignore"):
                x = fn(v) if v is not None and np.all(np.isfinite(v)) else np.nan
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            x = np.nan
        out.append(float(x))
    return out[0], out[1]


def failed_gates(rs, corr, mres, mstep, ctl, metric_applies):
    """Names of the acceptance gates that fail (rcond is enforced by factorize).

    This one function is the acceptance rule of every route; NaN never passes."""
    bad = []
    if not (np.isfinite(rs) and rs <= ctl.tol_scaled):
        bad.append("coord_residual")
    if not (np.isfinite(corr) and corr <= ctl.tol_step):
        bad.append("coord_step")
    if metric_applies:
        if not (np.isfinite(mres) and mres <= ctl.tol_metric_residual):
            bad.append("metric_residual")
        if not (np.isfinite(mstep) and mstep <= ctl.tol_metric_step):
            bad.append("metric_step")
    return tuple(bad)


@dataclass
class Certificate:
    ok: bool
    status: str
    resid_scaled: float
    resid_merit: float
    newton_corr_norm: float
    rcond: float
    evaluation: object = None
    factor: Factor | None = None
    message: str = ""
    metric_residual: float = np.nan
    metric_step: float = np.nan
    metric_scope: str = ""
    failed_gates: tuple = ()
    step: np.ndarray | None = None  # exact Newton correction p = H^{-1} S at this point


def _resid_merit(merit, S):
    if merit is None:
        return np.nan
    try:
        with np.errstate(all="ignore"):
            return float(np.sqrt(2.0 * merit(S)))
    except (FloatingPointError, ValueError):
        return np.nan


def certify(evaluate, theta, ctl, merit=None, metric_scope=None):
    """Fresh original-equation evaluation and root acceptance checks.

    With a fixed ``merit`` every gate of ACCEPTANCE_RULE_VERSION applies; without
    one (objective-only solves) only the coordinate/rcond gates apply and the
    certificate's ``metric_scope`` records that explicitly."""
    scope = METRIC_SCOPE_FIXED if merit is not None else (metric_scope or METRIC_SCOPE_OBJECTIVE)
    try:
        ev = evaluate(theta, True)
    except (FloatingPointError, ValueError, OverflowError) as exc:
        return Certificate(False, "numerical_failure", np.nan, np.nan, np.nan, np.nan, message=str(exc),
                           metric_scope=scope)
    S = ev.S
    if not np.all(np.isfinite(S)):
        return Certificate(False, "numerical_failure", np.nan, np.nan, np.nan, np.nan, ev, None, "non-finite score",
                           metric_scope=scope)
    rs = _scaled(S, ev.scale)
    rm = _resid_merit(merit, S)
    mres, _ = metric_values(merit, S, None)
    try:
        fac = factorize(ev.H, ctl.rcond_min)
    except SingularJacobian as exc:
        return Certificate(False, "singular_jacobian", rs, rm, np.nan, exc.rcond, ev, None, str(exc),
                           metric_residual=mres, metric_scope=scope)
    except FloatingPointError as exc:
        return Certificate(False, "numerical_failure", rs, rm, np.nan, np.nan, ev, None, str(exc),
                           metric_residual=mres, metric_scope=scope)
    step = fac.solve(S)
    corr = float(np.max(np.abs(step)))
    mres, mstep = metric_values(merit, S, step)
    bad = failed_gates(rs, corr, mres, mstep, ctl, merit is not None)
    ok = not bad
    return Certificate(ok, "ok" if ok else "root_check_failed", rs, rm, corr, fac.rcond, ev, fac,
                       "" if ok else "failed gates: " + ",".join(bad), mres, mstep, scope, bad, step)


def admit_with_metric(cert, merit, ctl):
    """Second-stage certification of an objective-only candidate (Gamma pilot).

    Uses the SAME evaluation and exact Newton correction as the coordinate
    certificate (no new fit, no new evaluation) and the fixed metric built from
    that candidate's own curvature (or the declared fallback).  Returns a new
    Certificate whose ``ok`` requires every gate of ACCEPTANCE_RULE_VERSION."""
    if cert is None or cert.factor is None or cert.evaluation is None:
        return Certificate(False, "root_check_failed", np.nan, np.nan, np.nan, np.nan,
                           message="no coordinate certificate to admit", metric_scope=METRIC_SCOPE_PILOT_ADMISSION)
    S = cert.evaluation.S
    step = cert.step if cert.step is not None else cert.factor.solve(S)
    mres, mstep = metric_values(merit, S, step)
    bad = failed_gates(cert.resid_scaled, cert.newton_corr_norm, mres, mstep, ctl, True)
    ok = cert.ok and not bad
    return Certificate(ok, "ok" if ok else "root_check_failed", cert.resid_scaled, _resid_merit(merit, S),
                       cert.newton_corr_norm, cert.rcond, cert.evaluation, cert.factor,
                       "" if ok else "failed gates: " + ",".join(bad or ("coordinate_certificate",)),
                       mres, mstep, METRIC_SCOPE_PILOT_ADMISSION, bad, step)


@dataclass
class Terminal:
    """Diagnostics of the LAST iterate a solve evaluated (accepted or not).

    For a failed solve these describe where the solver stopped; they are never
    a certificate.  rcond is the estimate that triggered a singular_jacobian
    exit when that is the cause."""

    resid_scaled: float = np.nan
    resid_merit: float = np.nan
    newton_corr_norm: float = np.nan
    rcond: float = np.nan
    beta: float = np.nan
    theta_max_abs: float = np.nan
    metric_residual: float = np.nan
    metric_step: float = np.nan


@dataclass
class SolveResult:
    theta: np.ndarray
    status: str
    iterations: int
    backtracks: int
    certificate: Certificate | None
    start_label: str
    elapsed_s: float
    history: list = field(default_factory=list)
    message: str = ""
    terminal: Terminal = field(default_factory=Terminal)
    certified_full_steps: int = 0  # final Newton steps accepted by a fresh certificate
    metric_scope: str = ""
    # optional: diagnostics of a full-step candidate whose certificate FAILED
    # after the line search failed (the returned iterate's terminal is unchanged)
    rejected_full_step: dict | None = None

    @property
    def ok(self):
        return self.status == "ok"


def _scaled(S, scale):
    # An infinite denominator would turn a finite nonzero residual into zero.
    # Invalid scaling inputs must reject on every shared certificate route.
    if not (np.all(np.isfinite(scale)) and np.all(scale > 0)):
        return np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.max(np.abs(S) / scale))


def _cert_diag(c):
    return {"status": c.status, "resid_scaled": c.resid_scaled, "newton_corr_norm": c.newton_corr_norm,
            "rcond": c.rcond, "metric_residual": c.metric_residual, "metric_step": c.metric_step,
            "failed_gates": list(c.failed_gates), "message": c.message}


def newton_solve(evaluate, theta0, ctl, merit=None, objective=None, start_label="", metric_scope=None):
    """Solve S(theta) = 0 from an explicit finite start.

    ``evaluate(theta, jacobian)`` returns an operators.Evaluation.  Exactly one
    of ``merit`` (fixed residual merit) or ``objective`` (convex potential with
    grad = -S) globalises the step.

    Line search: Armijo on the DIFFERENCE f(trial) - f(theta) <= c alpha slope,
    so a trial whose value rounds to f(theta) is never accepted as progress, and
    a trial equal to theta in floating point ends the search.  With an objective
    the Newton step must be a descent direction (slope = -S'p < 0) before any
    trial is evaluated.  When every backtrack fails, the full Newton step is
    accepted only if a fresh original-equation certificate at theta + p passes
    every unchanged root threshold (residual, Newton correction, rcond); this
    terminates at a certified root and is never used to continue iterating.

    Every acceptance route (the initial/iterative stop, the final fresh
    certificate and the full-step route) uses ``failed_gates``: with a merit
    the coordinate AND fixed-metric gates; with an objective the coordinate
    gates only (``metric_scope`` labels which).
    """
    if (merit is None) == (objective is None):
        raise ValueError("pass exactly one of merit or objective")
    scope = METRIC_SCOPE_FIXED if merit is not None else (metric_scope or METRIC_SCOPE_OBJECTIVE)
    rejected = None
    t0 = time.perf_counter()
    theta = np.array(theta0, dtype=float)
    if not np.isfinite(theta).all():
        raise ValueError("start must be finite")
    status, msg, iters, nback, hist = "max_iter", "", 0, 0, []
    term = Terminal()
    cert, certified = None, 0
    for it in range(ctl.max_iter + 1):
        term = Terminal(beta=float(theta[0]), theta_max_abs=float(np.max(np.abs(theta))))
        try:
            ev = evaluate(theta, True)
        except (FloatingPointError, ValueError, OverflowError) as exc:
            status, msg = "numerical_failure", str(exc)
            break
        S = ev.S
        if not np.all(np.isfinite(S)):
            status, msg = "numerical_failure", "non-finite score"
            break
        rs = _scaled(S, ev.scale)
        term.resid_scaled = rs
        term.resid_merit = _resid_merit(merit, S)
        term.metric_residual, _ = metric_values(merit, S, None)
        try:
            fac = factorize(ev.H, ctl.rcond_min)
        except SingularJacobian as exc:
            term.rcond = exc.rcond
            if exc.factor is not None:  # diagnostic |H^{-1} S| only; never a step
                with np.errstate(all="ignore"):
                    pd = exc.factor.solve(S)
                    term.newton_corr_norm = float(np.max(np.abs(pd)))
                term.metric_step = metric_values(merit, S, pd)[1]
            status, msg = "singular_jacobian", str(exc)
            break
        except FloatingPointError as exc:
            status, msg = "numerical_failure", str(exc)
            break
        p = fac.solve(S)  # H p = S  <=>  DS p = -S
        corr = float(np.max(np.abs(p)))
        term.newton_corr_norm, term.rcond = corr, fac.rcond
        term.metric_residual, term.metric_step = metric_values(merit, S, p)
        hist.append((it, rs, corr, fac.rcond, term.metric_residual, term.metric_step))
        # coordinate success alone never stops iteration while a metric gate fails
        if not failed_gates(rs, corr, term.metric_residual, term.metric_step, ctl, merit is not None):
            status = "ok"
            break
        if it == ctl.max_iter:
            status = "max_iter"
            break
        try:
            if merit is not None:
                f0 = merit(S)
                slope = -2.0 * f0
            else:
                f0 = objective(theta)
                slope = -float(S @ p)  # grad Q' p = -S'p
        except (FloatingPointError, ValueError, OverflowError) as exc:
            status, msg = "numerical_failure", str(exc)
            break
        if not (np.isfinite(f0) and np.isfinite(slope)):
            status, msg = "numerical_failure", "non-finite merit/objective or slope"
            break
        if not slope < 0:
            status, msg = "line_search_failed", (
                "Newton step is not a descent direction for Q" if merit is None
                else "zero residual merit without a certified root")
            break
        alpha, accepted, no_progress = 1.0, False, False
        for b in range(ctl.max_backtracks + 1):
            trial = theta + alpha * p
            if np.array_equal(trial, theta):
                no_progress = True
                nback += b
                break
            try:
                if merit is not None:
                    ft = merit(evaluate(trial, False).S)
                else:
                    ft = objective(trial)
                # strict decrease as well: guards an underflowed (-0.0) right side
                if np.isfinite(ft) and ft < f0 and (ft - f0) <= ctl.armijo * alpha * slope:
                    accepted = True
                    nback += b
                    break
            except (FloatingPointError, ValueError, OverflowError):
                pass
            alpha *= ctl.backtrack_factor
        if not accepted:
            if not no_progress:
                nback += ctl.max_backtracks
            full = certify(evaluate, theta + p, ctl, merit, scope)
            if full.ok:
                theta, cert, certified = theta + p, full, 1
                iters = it + 1
                status = "ok"
                break
            rejected = _cert_diag(full)  # recorded separately; terminal stays at theta
            status, msg = "line_search_failed", (
                "step below floating-point resolution of theta" if no_progress
                else f"no Armijo step after {ctl.max_backtracks} backtracks")
            break
        theta = trial
        iters = it + 1
    if status == "ok":
        if cert is None:
            cert = certify(evaluate, theta, ctl, merit, scope)
        if not cert.ok:
            status, msg = "root_check_failed", f"fresh certificate {cert.status}: {cert.message}"
        term = Terminal(cert.resid_scaled, cert.resid_merit, cert.newton_corr_norm, cert.rcond,
                        float(theta[0]), float(np.max(np.abs(theta))), cert.metric_residual, cert.metric_step)
    if status != "ok":
        cert = None  # a failed solve never carries a certificate
    return SolveResult(theta, status, iters, nback, cert, start_label, time.perf_counter() - t0, hist,
                       msg, term, certified, scope, rejected)
