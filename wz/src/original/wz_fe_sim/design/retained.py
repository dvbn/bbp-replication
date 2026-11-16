"""Sparse-local, ordinarily pair-centred retained design.

Coordinates theta = (beta, a_1..a_No, c_1..c_{Nd-1}) with blocks of width T-1,
so d = 1 + (No + Nd - 1)(T - 1).  The row of pair g=(i,j) at time t is

    b_gt = [x_gt - xbar_g,  F_t in block(origin i),  F_t in block(dest j)]

with F a fixed orthonormal T x (T-1) basis, F'1 = 0 (so the fixed-effect part is
already pair-centred).  One destination block (default: highest destination
position) is dropped.  Each pair touches at most K = 2T-1 coordinates.

Storage is per pair: ``bloc`` (G, T, K) holds the local T x K block B_g and
``coords`` (G, K) its global coordinates.  The dropped destination's local
columns are zero and point at a padding slot ``d`` that is discarded after
every scatter, so all pairs share one vectorised layout.  The n x d design is
never formed except by ``dense_design_for_tests`` (size-guarded).
"""

from __future__ import annotations

import numpy as np

DENSE_TEST_LIMIT = 5_000_000  # max n*d entries for the test-only dense design


def contrast_basis(T, kind="helmert"):
    """Orthonormal T x (T-1) time-contrast basis with F'1 = 0.

    ``helmert``: normalised Helmert columns (1,..,1,-k,0,..)/sqrt(k(k+1)).
    ``dct``: orthonormal DCT-II columns k = 1..T-1.
    An explicit array is validated and returned (used by invariance checks).
    """
    if not isinstance(kind, str):
        F = np.array(kind, dtype=float)
    elif kind == "helmert":
        F = np.zeros((T, T - 1))
        for k in range(1, T):
            F[:k, k - 1] = 1.0
            F[k, k - 1] = -float(k)
            F[:, k - 1] /= np.sqrt(k * (k + 1.0))
    elif kind == "dct":
        t = np.arange(T)[:, None] + 0.5
        k = np.arange(1, T)[None, :]
        F = np.sqrt(2.0 / T) * np.cos(np.pi * t * k / T)
    else:
        raise ValueError(f"unknown contrast basis {kind!r}")
    if F.shape != (T, T - 1):
        raise ValueError("contrast basis must be T x (T-1)")
    if np.max(np.abs(F.T @ F - np.eye(T - 1))) > 1e-12:
        raise ValueError("contrast basis must be orthonormal")
    if np.max(np.abs(F.sum(axis=0))) > 1e-12:
        raise ValueError("contrast basis must satisfy F'1 = 0")
    F.setflags(write=False)
    return F


class RetainedDesign:
    """Local pair blocks of the retained design for one (sub)panel.

    Parameters
    ----------
    x : (G, T) regressor by pair and period.
    pair_o, pair_d : (G,) origin / destination *positions* (0..No-1, 0..Nd-1).
    No, Nd : number of origin / destination positions (each must be used).
    F : contrast basis (T, T-1).
    dropped_dest : destination position whose block is dropped (default Nd-1).
    """

    def __init__(self, x, pair_o, pair_d, No, Nd, F, dropped_dest=None):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError("x must be (G, T) with T >= 2")
        G, T = x.shape
        pair_o = np.array(pair_o, dtype=np.int64)  # copies: caller arrays stay writable
        pair_d = np.array(pair_d, dtype=np.int64)
        if pair_o.shape != (G,) or pair_d.shape != (G,):
            raise ValueError("pair index arrays must have one entry per pair")
        if np.any((pair_o < 0) | (pair_o >= No)) or np.any((pair_d < 0) | (pair_d >= Nd)):
            raise ValueError("pair positions out of range")
        if len(np.unique(pair_o * Nd + pair_d)) != G:
            raise ValueError("duplicate pairs")
        if len(np.unique(pair_o)) != No or len(np.unique(pair_d)) != Nd:
            raise ValueError("every origin and destination position needs a pair")
        if not np.isfinite(x).all():
            raise ValueError("x must be finite")
        F = contrast_basis(T, F)
        dropped = Nd - 1 if dropped_dest is None else int(dropped_dest)
        if not 0 <= dropped < Nd:
            raise ValueError("dropped destination out of range")
        p = T - 1
        d = 1 + (No + Nd - 1) * p
        K = 1 + 2 * p
        dest_block = np.arange(Nd) - (np.arange(Nd) > dropped)
        dest_block[dropped] = -1
        keep = dest_block[pair_d] >= 0
        coords = np.empty((G, K), dtype=np.int64)
        coords[:, 0] = 0
        coords[:, 1 : 1 + p] = 1 + pair_o[:, None] * p + np.arange(p)[None, :]
        dcols = 1 + No * p + np.maximum(dest_block[pair_d], 0)[:, None] * p + np.arange(p)
        coords[:, 1 + p :] = np.where(keep[:, None], dcols, d)  # pad slot = d
        xc = x - x.mean(axis=1, keepdims=True)  # fixed ordinary pair centring
        bloc = np.zeros((G, T, K))
        bloc[:, :, 0] = xc
        bloc[:, :, 1 : 1 + p] = F[None, :, :]
        bloc[:, :, 1 + p :] = np.where(keep[:, None, None], F[None, :, :], 0.0)
        self.G, self.T, self.p, self.K, self.d = G, T, p, K, d
        self.No, self.Nd, self.dropped_dest = No, Nd, dropped
        self.pair_o, self.pair_d, self.keep_dest = pair_o, pair_d, keep
        self.F, self.xc, self.bloc, self.coords = F, xc, bloc, coords
        # flat (d+1)^2 indices of every local K x K entry, reused by every scatter
        self._flat = (coords[:, :, None] * (d + 1) + coords[:, None, :]).ravel()
        for arr in (pair_o, pair_d, keep, xc, bloc, coords, self._flat):
            arr.setflags(write=False)

    # ---- index and local <-> global maps ---------------------------------
    @property
    def n_obs(self):
        return self.G * self.T

    def gather_vec(self, v):
        """(G, K) local slices of a global d-vector (padding slot = 0)."""
        ext = np.zeros(self.d + 1)
        ext[: self.d] = v
        return ext[self.coords]

    def gather_mat(self, M):
        """(G, K, K) local blocks M[coords_g, coords_g] (padding rows/cols = 0)."""
        ext = np.zeros((self.d + 1, self.d + 1))
        ext[: self.d, : self.d] = M
        return ext.ravel()[self._flat].reshape(self.G, self.K, self.K)

    def scatter_vec(self, loc):
        """sum_g of local K-vectors into a global d-vector."""
        out = np.bincount(self.coords.ravel(), weights=loc.ravel(), minlength=self.d + 1)
        return out[: self.d]

    def scatter_mat(self, loc):
        """sum_g of local K x K blocks into a global d x d matrix."""
        out = np.bincount(self._flat, weights=loc.ravel(), minlength=(self.d + 1) ** 2)
        return out.reshape(self.d + 1, self.d + 1)[: self.d, : self.d]

    def index(self, theta):
        """q_g = B_g theta for every pair, (G, T)."""
        loc = self.gather_vec(theta)
        return np.einsum("gtk,gk->gt", self.bloc, loc)

    def apply_t(self, rho):
        """Local B_g' rho_g, (G, K)."""
        return np.einsum("gtk,gt->gk", self.bloc, rho)

    def sandwich_local(self, M):
        """Local B_g' M_g B_g for (G, T, T) M, (G, K, K)."""
        return np.matmul(self.bloc.transpose(0, 2, 1), np.matmul(M, self.bloc))

    def project_local(self, Mloc):
        """B_g Mloc_g B_g' for (G, K, K) local blocks, (G, T, T)."""
        return np.matmul(self.bloc, np.matmul(Mloc, self.bloc.transpose(0, 2, 1)))

    def gram(self):
        """B'B (d x d), used only for the documented fallback merit metric."""
        return self.scatter_mat(np.matmul(self.bloc.transpose(0, 2, 1), self.bloc))

    def dense_design_for_tests(self):
        """Explicit n x d design (rows ordered pair-major).  Tests only."""
        if self.n_obs * self.d > DENSE_TEST_LIMIT:
            raise MemoryError("dense design is test-only and size-guarded")
        Z = np.zeros((self.G, self.T, self.d + 1))
        g = np.arange(self.G)[:, None, None]
        t = np.arange(self.T)[None, :, None]
        np.add.at(Z, (g, t, self.coords[:, None, :]), self.bloc)
        return Z[:, :, : self.d].reshape(self.n_obs, self.d)


def retained_dimension(No, Nd, T):
    return 1 + (No + Nd - 1) * (T - 1)
