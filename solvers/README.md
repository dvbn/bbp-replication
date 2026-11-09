# PPML/GPML solver simulations

This directory contains the historical numerical implementations used for the paper's solver comparison. It is independent of the powermean package.

The study has 16 ordinary and four stress designs, sizes `(n,K)=(1000,10)` and `(10000,100)`, and 1,000 draws per cell: 40,000 datasets and 480,000 scheduled method/equation records. Certified structural cases remain as zero-fit placeholders. Unresolved cases are attempted; draws are never replaced or trimmed.

The six methods are iOLS with NumPy or C++, statsmodels Newton and IRLS, safeguarded scikit-learn Newton, and SciPy L-BFGS-B. PPML uses tau=1; GPML uses the log-link relative-residual quasi-score at tau=2, including zero outcomes. The iOLS start is its internal `log1p(y)` OLS start; the other methods start at zero slopes and `log(mean(y))` intercept. Controls and seeds are in `config.json`.

## Build and run

Install the repository's numerical/build requirements, a C++17 compiler, CMake and Eigen headers. Eigen 3.4 or later is required. Then, from the repository root:

```bash
python solvers/build_native.py --eigen-include /path/to/eigen3
python solvers/run.py --mode smoke --output outputs/solvers
python solvers/run.py --mode full --output outputs/full/solvers
```

The native build uses optimized Release settings and precise floating-point options. The C++ arm requires a built native module and never silently uses NumPy. `--native PATH` selects an explicit compatible module. The numerical runner uses POSIX process timers and is intended for Linux and macOS.

Use a fresh output directory for a separate run. `--resume` continues results with the same numerical source, configuration, dependency versions, Python ABI, actual native binary, budget policy and requested run scope. Changing the mode, methods, cells or replication count requires a new output directory. `--replications N` selects the first N original draw IDs; `--cells 'G_poisson_n1000_k10'` or comma-separated glob patterns selects cells. `--methods iols_numpy,statsmodels_newton` selects methods explicitly. These smaller runs demonstrate the numerical workflow; the complete paper experiment requires the full schedule.

## Timing and acceptance

The default `--budget historical` preserves the selected experiment's timing policy: most original calls had numerical iteration/evaluation limits without a process timer; 53,536 selected completion calls used 120 CPU and 300 wall second limits. `execution_schedule.json` lists those draw-ID ranges. `--budget common` applies process limits to every call; `--budget uncapped` removes those process limits. These are explicit alternative experiments and use separate resumption identities.

Raw results retain returned roots, native statuses, failures and measured clock overshoots. Common-budget eligibility is a separate field based on the observed CPU/wall times. The main table's descriptive median/mean milliseconds clip each recorded CPU time at 120 seconds across attempted fits; this is separate from a native stopping rule or an idealized timeout cost. New hardware, libraries and execution history will produce different timings.

The independent binary64 checker preserves the original strict 1e-8 and secondary 1e-6 score/fitted-index-correction gates. Native convergence alone is insufficient. The completed paper's subsequent numerical review retained 200 index-diagnostic qualifications: 198 score-rejected rows and two rows whose classifications were verified with high precision although an index scalar was not reproduced. These dispositions did not change either acceptance flag. The supplied saved summaries preserve those completed results; a fresh numerical run applies the original checker and does not claim to repeat the entire historical review process.

The revised iOLS curvature-grid policy was selected after inspecting 20 datasets. Their keys are in `config.json`; the run does not describe that selection as untouched independent confirmation. The C++ source is retained, while a fresh platform build has its own binary identity.

## Outputs

`fits.jsonl` stores full coefficient vectors, target estimates, statuses, raw clocks, independent checks and common-budget flags. `run.json` describes the selected scientific run. `summary/` contains fresh `paper_main_summary.csv`, `paper_accuracy_summary.csv`, `outcomes.csv` and `statistics.csv` for the repository's renderer. It uses these newly generated files when rebuilding displays after a full run. Regenerate summaries without refitting with:

```bash
python solvers/summarize.py --output outputs/full/solvers
python solvers/agreement.py --output outputs/full/solvers
```

The agreement step regenerates each dataset's design and compares all stored method pairs within an equation, including missing/nonfinite-vector states. It reports original-coordinate coefficient and fitted-index distances at both gates, in raw/common-budget views, with the separate exclusion of the 20 inspected datasets. A complete six-method run has 1.2 million dataset/equation/method-pair comparisons. A temporary disk index bounds memory; only the final CSV/JSON summaries remain. No new agreement tolerance or equality between PPML and GPML is asserted.

The supplied `saved/solvers/` files rebuild the published displays directly. They are intentionally compact scientific reporting inputs and contain no development databases or private review records. Appendix quantiles use the inverse empirical CDF; the main table uses its stated median/mean convention. Failures and structural exclusions remain in their original denominators.
