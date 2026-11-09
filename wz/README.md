# WZ fixed-effect simulations

This directory contains the Python implementation used for the paper's balanced three-way simulations, including cancelled GPML/PPML, analytical coefficient corrections, WZ PPML corrections and the separate covariance pilot. NumPy and SciPy are required. Numerical runs use one BLAS thread per worker and currently support Linux and macOS.

From the repository root:

```bash
python wz/run.py --mode smoke --output outputs/wz-smoke
python wz/run.py --mode full --output outputs/wz
```

The smoke runs DGP1/N20/T2/replication1, plus DGP1/N50/T2/replication1 for the covariance pilot. It checks execution, not Monte Carlo precision. Use `--skip-pilot` for just the main estimator example. A full run uses all four DGPs, all nine original base cells and the independent N500/T2 extension, each with 5,000 replications, plus the first 100 base replication IDs for the covariance pilot. This is substantial computation. `--paper-only` restricts the main run to the T2 cells used in the current displays; master dimensions and seed streams remain unchanged.

For a bounded run or separate pilot:

```bash
python wz/run.py --mode full --replications 2 --paper-only --skip-pilot --output outputs/wz-small
python wz/run.py --mode full --pilot-only --output outputs/wz-pilot
```

Use `--dgps 1,2`, `--cells 20x2,50x2`, and `--workers 2` to select original designs and worker count. These selectors change run coverage and must be stated when reporting results. `--replications R` always selects original IDs 1 through R, rather than redefining the master stream. `--pilot-replications R` caps the pilot at at most 100. Interruptions can be resumed with the same command plus `--resume`; different configurations, sources or NumPy/SciPy versions require a fresh output directory. An output directory permits one writer. A partial final JSONL line is reported for recovery, not silently discarded.

## Scientific settings

The base seed is 20261002, with a 100-by-100-by-10 master array per replication. The N500 extension uses independent seed 20261004 and a 500-by-500-by-2 master. The original Philox/SHA-256-keyed, raw-word Box–Muller generator and its block order are retained. Base DGPs and cells share each replication's master. The extension is not paired to the base across sample sizes. Diagonal pairs are included. The power grid is PPML/GPML only; jackknife estimates are disabled. Bootstrap summaries use 999 replication-bundle resamples and seed 20261003.

Two source directories preserve the historical numerical implementations. The paper retained the original successes for DGP1 at N20/N50, all original T values, and N100/T2 replications 1–3979. A refinement of Gamma-pilot admission was used for N100/T2 replication3980 and all remaining base cells, with the same equations and tolerances. The N500 extension and covariance pilot use that repaired implementation. The runner selects these versions explicitly; it does not replace the original successful estimates with an all-repaired rerun. Arithmetic and numerical branches are retained; only package initialization/configuration documentation and the surrounding public runner differ from the research trees.

The pilot holds the generic manuscript-corrected coefficient fixed when comparing ordinary, higher-order and WZ covariance. It uses the first 100 base innovations at N50/T2 in each DGP. This is separate from the 5,000-draw covariance comparison at the WZ-corrected PPML coefficient. Negative adjusted variances remain invalid intervals and are not clipped into validity.

## Outputs

The output folder contains `base/`, `n500/` and `pilot/`, plus `settings.json` with the scientific settings and actual dependency versions. Base/extension outputs are `summary.csv`, `methods.json`, and per-cell NPZ files with `<method>_beta` and `<method>_se` arrays. Replication-case JSONL files retain statuses and diagnostic quantities for resumption and independent aggregation; failed fits remain in denominators. These files contain newly computed results and do not depend on the supplied saved summaries.

Pilot outputs include `draw_results.csv`, `summary.csv` and `summary.json`. Completed roots are stored before covariance calculations so an inference failure cannot erase them. A covariance/root error stops the pilot and is recorded in `failures.jsonl`. Counts and exclusions should be checked before claiming a complete replication.

The repository's `saved/wz/` contains the completed paper results required for quick display reproduction. The GPML figure uses N20/50/100 at T2; the main table uses N50/100/500 at T2. PW uses the WZ coefficient correction with the ordinary pair sandwich. PWa uses the same corrected coefficient with WZ-adjusted covariance. These are distinct reporting choices.
