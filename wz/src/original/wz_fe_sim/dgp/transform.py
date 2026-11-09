"""Deterministic map: master innovations -> WZ Tables 1-3 design -> cell data.

Executable source facts (WZ public scripts; comments that disagree are wrong):
  alpha_it, gamma_jt, eta_ij ~ N(0, (1/16)^2)        rnormal(.,.,0,1/16)
  x_ij1 = alpha_i1 + gamma_j1 + nu_ij1,  x_ijt = x_ij,t-1/2 + alpha_it + gamma_jt + nu_ijt,
  nu ~ N(0, (1/2)^2); the pair effect does not enter x.
  lambda = exp(alpha + gamma + eta + beta0 x),  beta0 = 1
  z_ij1 = eps_ij1,  z_ijt = .3 z_ij,t-1 + sqrt(1-.09) eps_ijt   (Tables packet)
  u = exp(-log(1+c)/2 + sqrt(log(1+c)) z),  y = lambda u,  c = Var(y|.)/lambda^2:
     DGP1 c = lambda^-2, DGP2 c = lambda^-1, DGP3 c = 1,
     DGP4 c = .5 lambda^-1 + .5 exp(2x)   (Figure-1/historical variant: lambda^-1 + exp(2x))
Cells take the first N origins, first N destinations and first T periods of the
same master array (diagonal included), so all cells/DGPs of a replication share
random numbers and nothing depends on loop order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..design.cell import CellData


@dataclass(frozen=True)
class DGPParams:
    sd_effect: float = 1.0 / 16.0
    sd_nu: float = 0.5
    ar_rho: float = 0.3
    beta0: float = 1.0
    x_lag: float = 0.5
    dgp4_variant: str = "tables"  # "tables" (.5/.5) or "figure1" (1/1)

    @classmethod
    def from_config(cls, cfg):
        p = cfg["dgp_params"]
        return cls(float(p["sd_effect"]), float(p["sd_nu"]), float(p["ar_rho"]),
                   float(p["beta0"]), float(p["x_lag"]), str(p["dgp4_variant"]))


@dataclass(frozen=True)
class Master:
    x: np.ndarray  # (N, N, T)
    loglam: np.ndarray  # (N, N, T)
    z: np.ndarray  # (N, N, T)
    draw_hash: str
    params: DGPParams


def build_master(innov, params=DGPParams()):
    a = params.sd_effect * innov.alpha
    g = params.sd_effect * innov.gamma
    e = params.sd_effect * innov.eta
    nu = params.sd_nu * innov.nu
    N, _, T = innov.nu.shape
    x = np.empty((N, N, T))
    z = np.empty((N, N, T))
    x[:, :, 0] = a[:, None, 0] + g[None, :, 0] + nu[:, :, 0]
    z[:, :, 0] = innov.eps[:, :, 0]
    sq = np.sqrt(1.0 - params.ar_rho**2)
    for t in range(1, T):
        x[:, :, t] = params.x_lag * x[:, :, t - 1] + a[:, None, t] + g[None, :, t] + nu[:, :, t]
        z[:, :, t] = params.ar_rho * z[:, :, t - 1] + sq * innov.eps[:, :, t]
    loglam = a[:, None, :] + g[None, :, :] + e[:, :, None] + params.beta0 * x
    for arr in (x, z, loglam):
        arr.setflags(write=False)
    return Master(x, loglam, z, innov.draw_hash, params)


def variance_ratio(dgp, lam, x, variant="tables"):
    """c = Var(y | .)/lambda^2 for the four WZ variance designs."""
    if dgp == 1:
        return lam**-2.0
    if dgp == 2:
        return 1.0 / lam
    if dgp == 3:
        return np.ones_like(lam)
    if dgp == 4:
        if variant == "tables":
            return 0.5 / lam + 0.5 * np.exp(2.0 * x)
        if variant == "figure1":
            return 1.0 / lam + np.exp(2.0 * x)
        raise ValueError(f"unknown DGP4 variant {variant!r}")
    raise ValueError("dgp must be 1..4")


def outcome(loglam, x, z, dgp, variant="tables"):
    lam = np.exp(loglam)
    s2 = np.log1p(variance_ratio(dgp, lam, x, variant))
    return lam * np.exp(-0.5 * s2 + np.sqrt(s2) * z)


def make_cell(master, dgp, N, T, offdiag=False):
    """First N origins, first N destinations, first T periods; elementwise
    outcomes, so this equals slicing the full-master outcome array."""
    sl = (slice(0, N), slice(0, N), slice(0, T))
    x = master.x[sl]
    y = outcome(master.loglam[sl], x, master.z[sl], dgp, master.params.dgp4_variant)
    labels = tuple(range(1, N + 1))
    return CellData(x, y, labels, labels, offdiag)
