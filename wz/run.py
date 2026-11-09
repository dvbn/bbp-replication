"""Run the WZ simulations with the paper's generators and numerical routines."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"


@lru_cache(maxsize=2)
def backend(version):
    """Load each historical source version in its own module namespace."""
    name = "wz_" + version
    src = HERE / "src" / version / "wz_fe_sim"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, src / "__init__.py", submodule_search_locations=[str(src)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return {key: importlib.import_module(name + "." + rel) for key, rel in {
        "config": "config", "innovations": "dgp.innovations", "transform": "dgp.transform",
        "factorial": "estimators.factorial", "rows": "driver.rows", "tables": "summarise.tables",
        "correction": "estimators.correction", "wz": "estimators.wz_tau1",
    }.items()}


def clean(obj):
    """JSON preserves unavailable numeric quantities as null, never a fake zero."""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def original_source(config, campaign, dgp, n, t, rep):
    if campaign != "base" or dgp != config["source_selection"]["original_dgp"]:
        return False
    rule = config["source_selection"]
    if [n, t] in rule["original_complete_cells"]:
        return True
    p = rule["original_partial_cell"]
    return (n, t) == (p["N"], p["T"]) and p["first_rep"] <= rep <= p["last_rep"]


def source_digest():
    h = hashlib.sha256()
    for f in sorted((HERE / "src").rglob("*.py")):
        h.update(str(f.relative_to(HERE)).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    for name in ("run.py", "pilot_roots.py", "analytic_covariance.py"):
        h.update((HERE / name).read_bytes())
    return h.hexdigest()


def append_case(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(clean(record), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_cases(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.endswith("\n"):
                raise ValueError(f"Incomplete result line {i} in {path.name}; restore the interrupted line before resuming")
            rows.append(json.loads(line))
    reps = [int(r["rep"]) for r in rows]
    if len(reps) != len(set(reps)):
        raise ValueError(f"Duplicate replication in {path.name}")
    return rows


def case_file(output, campaign, dgp, n, t):
    return output / campaign / "cases" / f"dgp{dgp}_N{n}_T{t}.jsonl"


def fit_rep(task):
    """Generate a master once for the pending cells sharing its replication ID."""
    import numpy as np
    campaign, rep, cells, config = task
    repaired = backend("repaired")
    cfg_raw = config[campaign]
    cfg = repaired["config"].from_dict(cfg_raw)
    innov = repaired["innovations"].master_innovations(cfg["base_seed"], "M", rep, cfg["master_N"], cfg["master_T"])
    master = repaired["transform"].build_master(innov, repaired["transform"].DGPParams.from_config(cfg))
    records = []
    for dgp, n, t in cells:
        version = "original" if original_source(config, campaign, dgp, n, t, rep) else "repaired"
        mod = backend(version)
        local_cfg = mod["config"].from_dict(cfg_raw)
        begin = time.perf_counter()
        cell = mod["transform"].make_cell(master, dgp, n, t, bool(cfg["offdiag"]))
        error = ""
        try:
            fit = mod["factorial"].fit_cell(cell, local_cfg)
        except (FloatingPointError, ValueError, MemoryError, np.linalg.LinAlgError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            fit = mod["factorial"].failed_cell_fit(cell, local_cfg, error)
        base = dict(packet="M", archive_version=campaign, dgp=dgp, N=n, T=t, rep=rep,
                    offdiag=bool(cfg["offdiag"]), draw_hash=innov.draw_hash, draw_file_sha256="",
                    config_hash=mod["config"].config_hash(local_cfg), code_hash=version)
        rows = [r for r in mod["rows"].build_rows(fit, base) if r["estimator_version"] != "WZ-spj"]
        records.append(dict(campaign=campaign, dgp=dgp, N=n, T=t, rep=rep, source=version,
                            draw_hash=innov.draw_hash, elapsed_seconds=time.perf_counter()-begin,
                            error=error, rows=rows))
    return records


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def summarize_campaign(output, campaign, cells, config):
    import numpy as np
    tables = backend("repaired")["tables"]
    methods = config["methods"]
    ids = {(est, float(tau), coef, se): (mid, label) for mid, label, est, tau, coef, se in methods}
    needed = tuple(dict.fromkeys(tables.GROUP_FIELDS + ("rep", "beta", "se", "status_root", "status_inference",
                       "status_correction", "root_ambiguous", "draw_hash", "draw_file_sha256")))
    summaries = []
    folder = output / campaign
    folder.mkdir(exist_ok=True)
    for dgp, n, t in cells:
        records = sorted(read_cases(case_file(output, campaign, dgp, n, t)), key=lambda r: r["rep"])
        flat = [r for record in records for r in record["rows"]]
        if not flat:
            continue
        data = {key: np.asarray([np.nan if r[key] is None else r[key] for r in flat]) for key in needed}
        for key in ("packet", "archive_version", "estimator_version", "coef_version", "se_version", "draw_hash", "draw_file_sha256",
                    "status_root", "status_inference", "status_correction"):
            data[key] = data[key].astype(object)
        data["root_ambiguous"] = data["root_ambiguous"].astype(bool)
        group = next(iter(tables.build_cells(data).values()))
        arrays = {}
        for key, g in group.groups.items():
            mid, _ = ids[(key[5], float(key[6]), key[7], key[8])]
            arrays[mid + "_beta"] = g.beta
            arrays[mid + "_se"] = g.se
        tmp = folder / f"dgp{dgp}_N{n}_T{t}.npz.tmp"
        with tmp.open("wb") as f:
            np.savez_compressed(f, **arrays)
        tmp.replace(folder / f"dgp{dgp}_N{n}_T{t}.npz")
        rows = tables.summarise_cells(data, seed=config[campaign]["summary"]["bootstrap_seed"],
                                      n_boot=config[campaign]["summary"]["bootstrap_B"])
        for r in rows:
            r["method_id"], r["method_label"] = ids[(r["estimator_version"], float(r["tau"]), r["coef_version"], r["se_version"])]
        summaries.extend(rows)
    write_csv(folder / "summary.csv", summaries)
    (folder / "methods.json").write_text(json.dumps(methods, indent=2) + "\n")


def pilot_case(config, dgp, rep, root_path):
    import numpy as np
    mod = backend("repaired")
    sys.path.insert(0, str(HERE)) if str(HERE) not in sys.path else None
    from pilot_roots import reconstruct_cancelled, scalar_design, scalar_jets
    from analytic_covariance import covariance_refinement
    cfg = mod["config"].from_dict(config["base"])
    innov = mod["innovations"].master_innovations(cfg["base_seed"], "M", rep, cfg["master_N"], cfg["master_T"])
    master = mod["transform"].build_master(innov, mod["transform"].DGPParams.from_config(cfg))
    cell = mod["transform"].make_cell(master, dgp, 50, 2, False)
    design, y = cell.design(cfg["contrast_basis"]), cell.outcome_pairs()
    X = scalar_design(design)
    roots = reconstruct_cancelled(design, y, cfg, {})
    # Keep completed roots even if a subsequent covariance calculation fails.
    root_path.parent.mkdir(parents=True, exist_ok=True)
    with root_path.open("wb") as f:
        np.savez_compressed(f, theta_ppml=roots[1.].theta, theta_gpml_cancelled=roots[2.].theta)
    result = []
    for tau in (1., 2.):
        root = roots[tau]
        ev, fac = root.certificate.evaluation, root.certificate.factor
        inf = mod["correction"].pair_sandwich(design, ev, fac, cfg["tolerances"]["inference_rel"])
        if inf.status != "ok":
            raise RuntimeError(f"Pilot inference failed: {inf.status}")
        cor = mod["correction"].manuscript_correction(design, ev, fac, inf.adj)
        begin = time.perf_counter()
        ans = covariance_refinement(X, *scalar_jets(y, X @ root.theta, tau))
        beta = float(root.theta[0])
        row = dict(dgp=dgp, tau=tau, rep=rep, beta_raw=beta, beta_bc=beta-ans["coefficient_bias"],
                   **{k: ans[k] for k in ("coefficient_bias", "variance_raw", "variance_adjusted_raw", "delta",
                                         "delta_linear_second", "delta_linear_third", "delta_second_second", "positive_adjusted_variance")},
                   analytic_seconds=time.perf_counter()-begin,
                   variance_wz_unadjusted=None, variance_wz_adjusted=None, beta_wz_bc=None)
        np.testing.assert_allclose(ans["variance_raw"], inf.se**2, rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(ans["coefficient_bias"], cor.bias, rtol=2e-9, atol=2e-11)
        if tau == 1.:
            wz = mod["wz"].wz_correction(design, y, ev, cor.Hinv, n_countries=50)
            if wz.status_bias != "ok" or wz.status_cov != "ok":
                raise RuntimeError("Pilot WZ covariance comparison failed")
            row.update(variance_wz_unadjusted=wz.var_unadjusted, variance_wz_adjusted=wz.var_adjusted, beta_wz_bc=beta-wz.bias)
        result.append(row)
    return dict(rep=rep, dgp=dgp, draw_hash=innov.draw_hash, rows=result)


def summarize_pilot(output, config, dgps):
    import numpy as np
    folder = output / "pilot"
    raw = [r for dgp in dgps for rec in read_cases(folder / f"dgp{dgp}.jsonl") for r in rec["rows"]]
    write_csv(folder / "draw_results.csv", raw)
    summaries = []
    for dgp in dgps:
        for tau in (1., 2.):
            rows = [r for r in raw if r["dgp"] == dgp and r["tau"] == tau]
            if not rows:
                continue
            a = lambda key: np.asarray([np.nan if r[key] is None else r[key] for r in rows], dtype=float)
            mse = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)/np.sqrt(len(v))) if len(v)>1 else math.nan)
            beta, v0, v2 = a("beta_bc"), a("variance_raw"), a("variance_adjusted_raw")
            empirical = float(np.var(beta, ddof=1)) if len(beta)>1 else math.nan
            positive = np.isfinite(v2) & (v2 > 0)
            c0 = np.abs(beta-1) <= 1.96*np.sqrt(v0)
            c2 = positive & (np.abs(beta-1) <= 1.96*np.sqrt(np.maximum(v2, 0)))
            row = dict(dgp=dgp, tau=tau, R=len(rows), beta_bc_bias=float(beta.mean()-1),
                       beta_bc_sd_100=float(np.sqrt(empirical)), beta_bc_variance_100=empirical,
                       mean_variance_raw=mse(v0)[0], mcse_mean_variance_raw=mse(v0)[1],
                       mean_variance_new=mse(v2)[0], mcse_mean_variance_new=mse(v2)[1],
                       paired_mean_variance_change=mse(v2-v0)[0], mcse_paired_variance_change=mse(v2-v0)[1],
                       coverage_raw_at_beta_bc=mse(c0)[0], mcse_coverage_raw=mse(c0)[1],
                       coverage_new_at_beta_bc=mse(c2)[0], mcse_coverage_new=mse(c2)[1],
                       paired_coverage_change=mse(c2.astype(float)-c0)[0],
                       mcse_paired_coverage_change=mse(c2.astype(float)-c0)[1],
                       invalid_new_variances=int((~positive).sum()), ratio_of_mean_new_to_raw_variance=float(v2.mean()/v0.mean()),
                       mean_analytic_seconds=float(a("analytic_seconds").mean()))
            if tau == 1.:
                vw = a("variance_wz_adjusted")
                row.update(mean_variance_wz_unadjusted=float(a("variance_wz_unadjusted").mean()),
                           mean_variance_wz_adjusted=float(vw.mean()),
                           coverage_wz_variance_at_generic_beta_bc=float(np.mean(np.abs(beta-1)<=1.96*np.sqrt(vw))),
                           paired_new_minus_wz_variance=mse(v2-vw)[0], mcse_paired_new_minus_wz_variance=mse(v2-vw)[1],
                           mean_separate_wz_corrected_beta=float(a("beta_wz_bc").mean()))
            summaries.append(row)
    write_csv(folder / "summary.csv", summaries)
    (folder / "summary.json").write_text(json.dumps(clean(summaries), indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replications", type=int, help="First R of the original 5000 replication IDs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dgps", help="Comma-separated subset of 1,2,3,4")
    parser.add_argument("--cells", help="Comma-separated original cells, e.g. 20x2,50x2,500x2")
    parser.add_argument("--paper-only", action="store_true", help="Estimate T=2 cells only, preserving original master dimensions")
    parser.add_argument("--skip-pilot", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--pilot-replications", type=int, help="First R pilot IDs, at most 100")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads((HERE / "config.json").read_text())
    reps = args.replications if args.replications is not None else (1 if args.mode == "smoke" else 5000)
    pilot_reps = args.pilot_replications if args.pilot_replications is not None else min(100, reps)
    dgps = [int(x) for x in args.dgps.split(",")] if args.dgps else ([1] if args.mode == "smoke" else [1,2,3,4])
    all_cells = {tuple(c) for campaign in ("base", "n500") for c in config[campaign]["cells"]}
    cells = [tuple(map(int, x.lower().split("x"))) for x in args.cells.split(",")] if args.cells else (
        [(20,2)] if args.mode == "smoke" else sorted(all_cells))
    if args.paper_only:
        cells = [(n,t) for n,t in cells if t == 2]
    if not 1 <= reps <= 5000 or not 1 <= pilot_reps <= 100 or args.workers < 1:
        parser.error("Require 1 <= replications <= 5000, 1 <= pilot-replications <= 100 and workers >= 1")
    if not dgps or len(dgps) != len(set(dgps)) or not set(dgps) <= {1,2,3,4}:
        parser.error("dgps must be a duplicate-free subset of 1,2,3,4")
    if not cells or len(cells) != len(set(cells)) or not set(cells) <= all_cells:
        parser.error("cells must be a duplicate-free subset of the original design")
    if args.pilot_only and args.skip_pilot:
        parser.error("pilot-only and skip-pilot cannot both be selected")
    output = args.output.resolve()
    import numpy as np
    import scipy
    settings = dict(study="WZ", mode=args.mode, dgps=dgps, cells=[list(c) for c in cells], replications=reps,
                    pilot_replications=pilot_reps, pilot=not args.skip_pilot, pilot_only=args.pilot_only,
                    config=config, source_sha256=source_digest(), numpy=np.__version__, scipy=scipy.__version__)
    if output.exists() and any(output.iterdir()):
        if not args.resume or not (output / "settings.json").is_file():
            parser.error("Output is nonempty; choose a fresh directory or --resume an existing run")
        if json.loads((output / "settings.json").read_text()) != settings:
            parser.error("Cannot resume with different numerical settings, sources or dependency versions")
    output.mkdir(parents=True, exist_ok=True)
    # OS-managed lock is released even if a run is interrupted.
    import fcntl
    lock = (output / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another process is writing to this output directory")
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    if not args.pilot_only:
        for campaign in ("base", "n500"):
            selected = [(d,n,t) for d in dgps for n,t in cells if [n,t] in config[campaign]["cells"]]
            if not selected:
                continue
            done = {(d,n,t): {r["rep"] for r in read_cases(case_file(output,campaign,d,n,t))} for d,n,t in selected}
            tasks = [(campaign,r,[c for c in selected if r not in done[c]],config) for r in range(1,reps+1)]
            tasks = [task for task in tasks if task[2]]
            if args.workers == 1:
                results = map(fit_rep, tasks)
                pool = None
            else:
                pool = ProcessPoolExecutor(max_workers=args.workers)
                results = pool.map(fit_rep, tasks, chunksize=1)
            try:
                for records in results:
                    for rec in records:
                        append_case(case_file(output,campaign,rec["dgp"],rec["N"],rec["T"]),rec)
                        print(f"{campaign}: DGP{rec['dgp']} N{rec['N']} T{rec['T']} rep{rec['rep']} [{rec['source']}]", flush=True)
            finally:
                if pool is not None:
                    pool.shutdown()
            summarize_campaign(output,campaign,selected,config)
    if not args.skip_pilot:
        folder = output / "pilot"
        folder.mkdir(exist_ok=True)
        for dgp in dgps:
            path = folder / f"dgp{dgp}.jsonl"
            done = {r["rep"] for r in read_cases(path)}
            for rep in range(1,pilot_reps+1):
                if rep not in done:
                    try:
                        rec = pilot_case(config,dgp,rep,folder / "roots" / f"dgp{dgp}_rep{rep:06d}.npz")
                    except Exception as exc:
                        append_case(folder / "failures.jsonl", dict(dgp=dgp,rep=rep,error=f"{type(exc).__name__}: {exc}"))
                        raise
                    append_case(path,rec)
                    print(f"pilot: DGP{dgp} N50 T2 rep{rep}",flush=True)
        summarize_pilot(output,config,dgps)
    print(f"WZ {args.mode} complete: {output}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
