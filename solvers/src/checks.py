"""Independent exact-equation audit and existence screens (no fixed effects).

prepare(X, y, config) runs once per dataset, outside fit timing. It freezes the
QR coordinates X = QR, Z = sqrt(n) Q, and classifies each tau:
  eligible                    full rank and y > 0: these convex criteria are
                              coercive, so a finite unique root exists
  structural_rank_deficient   coefficients not unique; never redrawn or truncated
  structural_nonexistence     verified certificate (screen_details names it)
  unknown                     zeros present, no certificate; LP failure lands here
evaluate(context, beta, tau, config) audits any arm's beta against the original
equation: g = Z'r/n, H = Z'diag(c)Z/n, score_error = |g|, index_correction =
|H^-1 g| (= RMS of the Newton fitted-index correction because Z'Z/n = I).
  PPML r = (y - mu)/ybar, c = mu/ybar     GPML r = y/mu - 1, c = y/mu (y > 0 masks,
  y/mu = exp(log y - eta) so a huge y and a large index cannot overflow in a product)
Config: config['checks'][CHECKS_KEYS]. screen_status is keyed by tau (1, 2); the
driver maps each estimator to its config tau. Statuses are SCREEN_STATUSES.
HiGHS threading is not governed by the BLAS variables. These pinned SciPy calls
forward threads=1 and parallel=False to HiGHS; host verification inspected the
native options immediately before the actual LP solve. The expected SciPy
forwarding warning is retained in lp_warnings.
"""
import warnings

import numpy as np
from scipy.optimize import linprog

GATES = {'accepted_1e8': 1e-8, 'accepted_1e6': 1e-6}
CHECKS_KEYS = ('intercept_column', 'gpml_lp_min_slope', 'ppml_separation_lp')
SCREEN_STATUSES = ('eligible', 'unknown', 'structural_rank_deficient', 'structural_nonexistence')
TAUS = (1, 2)
EPS = np.finfo(float).eps


def _tau(tau):
    if tau not in TAUS:
        raise ValueError('tau must be 1 or 2')
    return int(tau)


def prepare(X, y, config):
    opts = {key: config['checks'][key] for key in CHECKS_KEYS}
    X = np.ascontiguousarray(X, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or len(y) != len(X):
        raise ValueError('X must be n x k and y length n')
    if not (np.isfinite(X).all() and np.isfinite(y).all() and (y >= 0).all()):
        raise ValueError('X and y must be finite and y nonnegative')
    n, k = X.shape
    Q, R = np.linalg.qr(X)
    singular = np.linalg.svd(R, compute_uv=False)
    rank = int((singular > singular.max() * max(n, k) * EPS).sum()) if singular.size else 0
    ic = opts['intercept_column']
    if ic is not None and not np.all(X[:, ic] == 1.0):
        ic = None
    # An exactly constant non-intercept column is exact collinearity, whatever the SVD tolerance says.
    constant = 0
    if ic is not None and n:
        constant = int(sum(bool(np.all(X[:, j] == X[0, j])) for j in range(k) if j != ic))
    full_rank = rank == k and constant == 0
    zeros = int((y == 0).sum())
    details = dict(n=n, k=k, rank=rank, constant_columns=constant, n_zero=zeros, intercept_column=ic,
                   cond_R=float(singular.max() / singular.min()) if singular.min() > 0 else float('inf'),
                   common_certificate=None)
    if not full_rank:
        status = {1: 'structural_rank_deficient', 2: 'structural_rank_deficient'}
    elif zeros == 0:
        status = {1: 'eligible', 2: 'eligible'}
    else:
        details['common_certificate'] = support_certificate(X, y, ic)
        if details['common_certificate']:
            status = {1: 'structural_nonexistence', 2: 'structural_nonexistence'}
        else:
            # A GPML recession certificate says nothing about PPML: tau 1 stays unknown.
            gpml = gpml_certificate(X, y, ic, opts)
            details.update(('gpml_' + key, value) for key, value in gpml.items())
            status = {1: 'unknown', 2: 'structural_nonexistence' if gpml['certified'] else 'unknown'}
            if opts['ppml_separation_lp']:
                details.update(('ppml_' + key, value) for key, value in ppml_separation_candidate(X, y).items())
    return dict(X=X, y=y, Q=Q, n=n, k=k, full_rank=full_rank, ybar=float(y.mean()),
                screen_status=status, screen_details=details)


def support_certificate(X, y, ic):
    """Obvious recession directions valid for both tau, or None."""
    if y.max() == 0:
        return 'all_outcomes_zero'
    for j in range(X.shape[1]):
        column = X[:, j]
        if j == ic or not np.all((column == 0) | (column == 1)) or column.min() == column.max():
            continue
        if y[column == 1].max() == 0:            # direction -e_j
            return 'all_zero_group_column%d_eq1' % j
        if ic is not None and y[column == 0].max() == 0:   # direction e_j - e_intercept
            return 'all_zero_group_column%d_eq0' % j
    return None


def gpml_certificate(X, y, ic, config):
    """GPML only. L = mean(y exp(-Xb) + Xb) is unbounded below along v if
    X_pos v >= 0 and mean(X v) < 0. The LP proposes v; acceptance needs a
    strictly positive margin and negative slope after explicit floating-point
    error bounds. No certificate never means existence."""
    out = dict(certified=False, lp_status=None, lp_value=None, margin=None, slope_upper=None,
               lp_warnings=None)
    if ic is None:
        out['lp_status'] = 'skipped_no_intercept'
        return out
    positive = y > 0
    P = X[positive]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        lp = linprog(X.mean(axis=0), A_ub=-P, b_ub=np.zeros(len(P)),
                     bounds=[(-1.0, 1.0)] * X.shape[1], method='highs',
                     options={'threads': 1, 'parallel': False})
    out.update(lp_status=int(lp.status), lp_warnings=len(caught))
    if not lp.success or not lp.fun < 0:
        return out
    out['lp_value'] = float(lp.fun)
    v = np.array(lp.x, dtype=float)
    # Shift the index by a constant so every positive row is strictly inside.
    v[ic] += max(0.0, -float(np.min(P @ v))) + 0.25 * abs(float(lp.fun))
    index = X @ v
    error = (X.shape[1] + 2) * EPS * (np.abs(X) @ np.abs(v))
    margin = float(np.min(index[positive] - error[positive]))
    slope_upper = float(np.mean(index + error) + len(y) * EPS * np.mean(np.abs(index)))
    out.update(margin=margin, slope_upper=slope_upper,
               certified=bool(margin > 0 and slope_upper < -config['gpml_lp_min_slope']))
    return out


def ppml_separation_candidate(X, y):
    """Diagnostic only: PPML separation needs X_pos v = 0 exactly, which a
    tolerance LP cannot verify. A candidate never changes screen_status."""
    positive = y > 0
    Z0 = X[~positive]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        lp = linprog(Z0.sum(axis=0), A_ub=Z0, b_ub=np.zeros(len(Z0)), A_eq=X[positive],
                     b_eq=np.zeros(int(positive.sum())), bounds=[(-1.0, 1.0)] * X.shape[1], method='highs',
                     options={'threads': 1, 'parallel': False})
    out = dict(lp_status=int(lp.status), lp_warnings=len(caught), lp_value=None,
               eq_residual=None, separation_candidate=False)
    if lp.success:
        out.update(lp_value=float(lp.fun), eq_residual=float(np.max(np.abs(X[positive] @ lp.x))),
                   separation_candidate=bool(lp.fun < -1e-9))
    return out


def evaluate(context, beta, tau, config):
    """Audit one beta against the original tau equation (no rank dropping, no clipping).

    Every early exit returns a complete record: curvature_status stays 'not_evaluated'.
    """
    tau = _tau(tau)
    out = dict(screen_status=context['screen_status'][tau], beta_status='ok',
               accepted_1e8=False, accepted_1e6=False, score_error=None,
               index_correction=None, curvature_status='not_evaluated', min_eig_H=None, cond_H=None,
               certificate_conflict=False)
    if beta is None:
        out['beta_status'] = 'missing'
        return out
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (context['k'],):
        raise ValueError('beta has shape %r, expected (%d,)' % (beta.shape, context['k']))
    if not np.isfinite(beta).all():
        out['beta_status'] = 'nonfinite'
        return out
    X, y, n = context['X'], context['y'], context['n']
    eta = X @ beta
    with np.errstate(over='ignore', invalid='ignore'):
        if tau == 1:
            mu = np.exp(eta)
            unit = context['ybar'] if context['ybar'] > 0 else 1.0   # frozen sample-mean units
            resid, curvature = (y - mu) / unit, mu / unit
        else:
            positive = y > 0
            ratio = np.zeros(n)
            ratio[positive] = np.exp(np.log(y[positive]) - eta[positive])
            resid, curvature = ratio - 1.0, ratio
    if not (np.isfinite(resid).all() and np.isfinite(curvature).all()):
        out['beta_status'] = 'index_overflow'
        return out
    Q = context['Q']
    g = Q.T @ resid / np.sqrt(n)                 # Z' r / n
    H = (Q.T * curvature) @ Q                    # Z' diag(c) Z / n
    H = 0.5 * (H + H.T)
    if not (np.isfinite(g).all() and np.isfinite(H).all()):
        out['beta_status'] = 'index_overflow'
        return out
    out['score_error'] = float(np.linalg.norm(g))
    eig = np.linalg.eigvalsh(H)
    out.update(min_eig_H=float(eig[0]), cond_H=float(eig[-1] / eig[0]) if eig[0] > 0 else float('inf'))
    if not context['full_rank']:
        out['curvature_status'] = 'rank_deficient_design'
    elif not np.isfinite(eig).all() or eig[0] <= context['k'] * EPS * eig[-1]:
        out['curvature_status'] = 'not_positive_definite'
    else:
        out['curvature_status'] = 'positive_definite'
        L = np.linalg.cholesky(H)
        step = np.linalg.solve(L.T, np.linalg.solve(L, g))
        out['index_correction'] = float(np.linalg.norm(step))
        for name, gate in GATES.items():
            out[name] = out['score_error'] <= gate and out['index_correction'] <= gate
    # Only reachable if a structural screen is ever fitted; the driver stops on it.
    out['certificate_conflict'] = bool(out['accepted_1e6'] and out['screen_status'].startswith('structural'))
    return out


def resolve_existence(screen_status, evaluations):
    """Report-level label for one dataset and tau, pooling evaluate() dicts over arms.

    A verified regular root (1e-8 gates, positive definite H) from any arm is
    numerical existence evidence for the shared estimator; it is not a proof.
    """
    if screen_status == 'unknown' and any(e['accepted_1e8'] for e in evaluations):
        return 'existence_evidenced_by_verified_root'
    return screen_status
