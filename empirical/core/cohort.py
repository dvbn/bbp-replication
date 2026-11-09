"""Duck-typed seam between unequal country groups and the unchanged WZ operators.

``wz_fe_sim.design.operators`` evaluates one balanced block of G groups of T
rows through ``design.bloc`` (G, T, K), ``index``, ``apply_t``, ``scatter_vec``
and ``scatter_mat``.  Here every country of size T goes into the cohort of
size T, every slope column is retained (K = d = p) and the global scatter is a
plain sum over the cohort's groups.  The panel evaluates the unchanged
``cancelled_eval`` / ``joint_eval`` on every cohort and sums S, H and scale.
Nothing is padded: a cohort holds exactly its countries' rows.
"""

from __future__ import annotations

import numpy as np

from wz_fe_sim.design import operators as ops


class Cohort:
    def __init__(self, rows, B, y):
        self.rows = np.array(rows, dtype=np.int64)  # (G, T) sample row indices
        self.bloc = np.array(B[self.rows], dtype=float)  # (G, T, p)
        self.y = np.array(y[self.rows], dtype=float)  # (G, T)
        self.G, self.T, self.p = self.bloc.shape
        self.d = self.K = self.p
        for arr in (self.rows, self.bloc, self.y):
            arr.setflags(write=False)

    @property
    def n_obs(self):
        return self.G * self.T

    def index(self, theta):
        return self.bloc @ np.asarray(theta, dtype=float)

    def apply_t(self, rho):
        return np.einsum("gtk,gt->gk", self.bloc, rho)

    def scatter_vec(self, loc):
        return loc.sum(axis=0)

    def scatter_mat(self, loc):
        return loc.sum(axis=0)

    def gather_vec(self, v):
        return np.broadcast_to(np.asarray(v, dtype=float), (self.G, self.p))

    def gram(self):
        return np.einsum("gtk,gtl->kl", self.bloc, self.bloc)


class Panel:
    """All cohorts of one sample; exposes the ``d``/``n_obs``/``gram`` used by
    the vendor's ``build_metric`` and the summed evaluators used by Newton."""

    def __init__(self, cohorts):
        self.cohorts = tuple(cohorts)
        self.d = self.cohorts[0].p
        self.n_obs = int(sum(c.n_obs for c in self.cohorts))

    def gram(self):
        return sum(c.gram() for c in self.cohorts)

    def evaluator(self, kind, tau):
        """``evaluate(theta, jacobian)`` of the summed cancelled ("C") or
        profiled joint ("J") equation; per-cohort states are kept in order."""
        fn = {"C": ops.cancelled_eval, "J": ops.joint_eval}[kind]
        tau = float(tau)

        def evaluate(theta, jacobian=True):
            parts = [fn(c, c.y, theta, tau, jacobian) for c in self.cohorts]
            S = sum(e.S for e in parts)
            scale = sum(e.scale for e in parts)
            H = sum(e.H for e in parts) if jacobian else None
            factor = tuple(e.factor for e in parts) if kind == "J" else None
            return ops.Evaluation(S, H, scale, tuple(e.state for e in parts), factor)

        return evaluate

    def objective(self, tau):
        """Summed convex cancelled potential (vendor: tau in {1, 2} only)."""
        return lambda theta: float(sum(ops.cancelled_objective(c, c.y, theta, tau) for c in self.cohorts))

    def states(self, theta, tau):
        return [ops.evaluate(c, c.y, theta, tau) for c in self.cohorts]


def build_panel(rows_by_cohort, B, y):
    return Panel(Cohort(r, B, y) for r in rows_by_cohort)
