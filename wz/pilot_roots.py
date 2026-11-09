"""Cancellation roots and scalar derivatives used by the covariance pilot."""
from __future__ import annotations
import numpy as np
from wz_repaired.design import operators as ops
from wz_repaired.estimators.factorial import gamma_pilot, build_metric
from wz_repaired.estimators.newton import SolverControl, admit_with_metric, newton_solve
from wz_repaired.driver.rows import solve_result_diagnostics

def reconstruct_cancelled(design, y, cfg, record):
    """Cancellation branch of archived fit_cell, using its unchanged helpers.

    Preserve Gamma objective start, fixed-metric admission/refinement, start
    order, exact duplicate-start reuse, first accepted root and ambiguity gate.
    Intentionally omit the unrelated joint-root and SPJ branches.
    """
    ctl = SolverControl.from_config(cfg)
    pilot = gamma_pilot(design, y, ctl)
    record["pilot_initial"] = solve_result_diagnostics(pilot)
    merit, source = build_metric(design, pilot)
    record["merit_source"] = source
    admission = admit_with_metric(pilot.certificate, merit, ctl) if pilot.ok and merit is not None else None
    if (admission is not None and not admission.ok and admission.failed_gates
            and set(admission.failed_gates) <= {"metric_residual", "metric_step"}):
        pilot = newton_solve(lambda th, jac: ops.cancelled_eval(design, y, th, 2., jac),
                             pilot.theta, ctl, merit=merit, start_label="gamma_metric_refinement")
        record["pilot_refinement"] = solve_result_diagnostics(pilot)
        admission = admit_with_metric(pilot.certificate, merit, ctl) if pilot.ok else None
    if merit is None or not pilot.ok or admission is None or not admission.ok:
        raise RuntimeError("No admitted Gamma pilot; no cancelled root attempted")
    record["metric"] = merit.describe()
    roots = {}
    record["root_attempts"] = {}
    neighbor = {float(k): None if v is None else float(v) for k, v in cfg["continuation_neighbor"].items()}
    for tau in map(float, cfg["cancel_solve_order"]):
        starts = [("gamma_pilot", pilot.theta)]
        nb = neighbor.get(tau)
        if nb is not None and nb in roots:
            starts.append((f"neighbor_tau{nb:g}", roots[nb].theta))
        if tau != 1. and 1. in roots:
            starts.append(("ppml_root", roots[1.].theta))
        done, solved, detail = [], [], []
        record["root_attempts"][str(tau)] = detail
        for label, theta in starts:
            duplicate = next((name for name, old in done if np.array_equal(old, theta)), None)
            if duplicate is not None:
                detail.append({"start": label, "status": f"reused:{duplicate}", "new_solve": False})
                continue
            result = newton_solve(lambda th, jac: ops.cancelled_eval(design, y, th, tau, jac),
                                  theta, ctl, merit=merit, start_label=label)
            done.append((label, theta))
            solved.append(result)
            detail.append({"new_solve": True, **solve_result_diagnostics(result)})
            if result.ok and not cfg["run_all_starts"]:
                break
        accepted = [r for r in solved if r.ok]
        if not accepted:
            raise RuntimeError(f"No accepted cancelled root tau={tau}")
        root = accepted[0]
        if max(abs(float(r.theta[0]-root.theta[0])) for r in accepted) > cfg["tolerances"]["tol_root"]:
            raise RuntimeError(f"Ambiguous cancelled roots tau={tau}")
        roots[tau] = root
    return roots

def scalar_design(design):
    Xpad = np.zeros((design.G, design.d + 1))
    np.add.at(Xpad, (np.arange(design.G)[:, None], design.coords), design.bloc[:, 0, :])
    X = Xpad[:, :design.d]
    if not np.allclose(design.bloc[:, 1, :], -design.bloc[:, 0, :], atol=1e-14, rtol=0):
        raise AssertionError("T=2 ordinary-centered coordinate identity failed")
    return X

def scalar_jets(y, q, tau):
    if tau == 1.:
        total = y.sum(axis=1)
        tanh = np.tanh(q)
        sech2 = 1 / np.cosh(q)**2
        f = y[:, 0] - y[:, 1] - total*tanh
        j = -total*sech2
        k = 2*total*sech2*tanh
        u = 2*total*sech2*(1-3*tanh*tanh)
    elif tau == 2.:
        pos, neg = y[:, 0]*np.exp(-q), y[:, 1]*np.exp(q)
        f, j = pos-neg, -(pos+neg)
        k, u = f.copy(), j.copy()
    else:
        raise ValueError("Only cancelled PPML and GPML authorized")
    return f, j, k, u
