# Solver comparison

This folder reproduces the solver comparison of Section 4.3 of *Dealing with Logs and Zeros in Regression Models* (Table 1) and its Supplement S5.1 (appendix tables, the accuracy figure and the agreement statistics). The study asks whether IOLS, the iterated ordinary least squares algorithm of Section 4, computes PPML and GPML estimates reliably and at competitive cost, compared with standard solvers applied to exactly the same estimating equations on exactly the same data.

## What is estimated

**Model and equations.** Every design satisfies $E(Y\mid X)=\exp(X'\beta_0)$ without fixed effects. Each dataset is fitted twice, by PPML ($\tau=1$) and by GPML ($\tau=2$), the solutions of

$$\frac{1}{n}\sum_{i=1}^n X_i\,\mu_i(\beta)^{1-\tau}\bigl(Y_i-\mu_i(\beta)\bigr)=0,\qquad \mu_i(\beta)=\exp(X_i'\beta).$$

PPML weights level residuals $Y_i-\mu_i$ and GPML weights relative residuals $Y_i/\mu_i-1$. Both equations admit zero outcomes.

**Methods.** The same two equations are solved by six methods:

| Method | Implementation | Starting value |
|---|---|---|
| IOLS (C++) | The paper's algorithm, compiled kernel (`native/`) | Internal OLS fit of $\log(1+y)$ |
| IOLS (NumPy) | The same algorithm in NumPy | Internal OLS fit of $\log(1+y)$ |
| Newton | statsmodels GLM, observed-Hessian Newton | Zero slopes, intercept $\log(\bar y)$ |
| IRLS | statsmodels GLM, Fisher scoring | Zero slopes, intercept $\log(\bar y)$ |
| Damped Newton | scikit-learn's Newton–Cholesky solver with backtracking line search | Zero slopes, intercept $\log(\bar y)$ |
| L-BFGS-B | SciPy | Zero slopes, intercept $\log(\bar y)$ |

Table 1 of the paper shows C++ IOLS, Newton and IRLS; the Supplement reports all six. For GPML, statsmodels receives an adapter that evaluates the quasi-log-likelihood $-\sum_i(Y_i/\mu_i+\log\mu_i)$ with a zero ratio term when $Y_i=0$; the package's score, Hessian and IRLS updates are unchanged.

**Designs.** In every design the treatment is an independent Bernoulli(1/2) variable with coefficient $\log(1.08)$, so the true effect on the mean is 8 percent, and the untreated population mean is 2. Sixteen ordinary designs combine outcome distributions (Poisson; Gamma; negative binomial with size 5, 1 or 0.2; compound Poisson-Gamma; inverse Gaussian; negative binomial with regressor-dependent size; lognormal; zero-inflated lognormal) with Gaussian, mixed, Gamma or rare-binary regressors. Four stress designs have high dispersion (negative binomial), heavy tails (lognormal), rare large outcomes (contamination with probability 0.001 and multiplier 500) or near collinearity (regressor correlation 0.999). Each design is drawn 1,000 times at $(n,K)=(1{,}000,10)$ and at $(10{,}000,100)$, where $K$ counts the intercept and the treatment: 40,000 datasets and 480,000 scheduled method-equation records. Supplement S5.1 gives every parameter; the code is in `src/designs.py`.

## How success is defined

**Accuracy.** A solver's own convergence report is not used. Every returned coefficient vector is checked against the original estimating equation by an independent checker: in QR-standardized coordinates, the Euclidean norm of the score and the root-mean-square local correction to the fitted index must both be at most $10^{-8}$, and the observed curvature must be numerically positive definite. A secondary check uses $10^{-6}$.

**Time budget.** A fit is successful under the common budget if it passes the accuracy check within 120 CPU seconds and 300 wall-clock seconds. Table 1 reports median and mean CPU times over all attempted fits, including failures, with each recorded CPU time capped at 120 seconds; the Supplement tables report the raw times. Setup time is excluded.

**Structural exclusions.** Before fitting, a screen classifies each dataset and equation. A rank-deficient design, or one for which a linear-programming certificate shows that no finite solution exists, is recorded as a structural exclusion. The dataset is never redrawn or trimmed, and excluded cases are left out of the attempted-fit denominators.

**Disclosed tuning and review.** The IOLS damping grid used in the reported runs was revised after inspecting 20 datasets of the reported sample. The paper reports this as a retrospective update rather than an untouched confirmation, and the agreement statistics are also computed without these datasets, whose keys are listed in [`config.json`](config.json). After the study was completed, a numerical review retained 200 index-diagnostic qualifications (198 rows rejected on the score and two rows whose classification was verified in high precision although an index scalar was not reproduced); they changed no acceptance flag. The saved summaries reflect the completed study; a fresh run applies the original checker but not that review.

## Build the C++ IOLS module

The C++ IOLS method needs a compiled module. Install CMake, a C++17 compiler, BLAS/LAPACK and the [Eigen](https://eigen.tuxfamily.org/) headers (version 3.4 or later), then run from the repository root, in the Python environment that will run the study:

```bash
python solvers/build_native.py --eigen-include /path/to/eigen3   # directory containing Eigen/Core
```

The build uses an optimized release configuration and precise floating-point options. The C++ method never falls back silently on NumPy: if the module is missing, a run that includes this method stops before fitting and asks for the build. `--native PATH` selects a specific compatible module, and `--methods` can exclude the C++ method.

## Run

```bash
python solvers/run.py --mode smoke --output outputs/solvers        # first design cell, one draw, six methods, two equations
python solvers/run.py --mode full --output outputs/full/solvers    # complete study
```

| Option | Effect |
|---|---|
| `--replications N` | Use the first N of the original 1,000 draws per cell |
| `--cells 'G_poisson_n1000_k10'` | Select cells by name or glob pattern (comma-separated) |
| `--methods iols_numpy,statsmodels_newton` | Select methods (default: all six) |
| `--budget historical` | Default; reproduces the time limits of the original study (below) |
| `--budget common` | Apply the 120/300-second process limits to every call |
| `--budget uncapped` | Remove process time limits |
| `--resume` | Continue an interrupted run |

**Time limits in the original study.** Most original calls ran with the solvers' iteration and evaluation limits but without a process timer; 53,536 completion calls ran under the 120 CPU and 300 wall-clock second limits. [`execution_schedule.json`](execution_schedule.json) lists those draws, and the default `--budget historical` reproduces this schedule. Whether a fit met the common budget is recorded separately, from its measured CPU and wall-clock times. The other two budgets define different experiments and are resumed separately.

Smaller runs show that the workflow runs; the paper's results require the complete schedule. Use a fresh output directory for each configuration. `--resume` continues only a run with the same numerical source, configuration, dependency versions, Python ABI, compiled module, budget and scope. The runner uses POSIX process timers and is intended for Linux and macOS. New hardware and libraries give different timings.

## Outputs

- `fits.jsonl`: one record per fit with the full coefficient vector, target estimates, solver status, CPU and wall times, the independent accuracy checks and the common-budget flag.
- `run.json`: the scientific settings of the run.
- `summary/`: `paper_main_summary.csv`, `paper_accuracy_summary.csv`, `outcomes.csv` and `statistics.csv`, which the reporting scripts use to rebuild the displays after a full run.

The summaries can be rebuilt from `fits.jsonl` without refitting:

```bash
python solvers/summarize.py --output outputs/full/solvers
python solvers/agreement.py --output outputs/full/solvers
```

The agreement step regenerates each dataset and compares the coefficient vectors and fitted indices of every pair of methods for the same equation, at both accuracy thresholds, with and without the common budget, and with and without the 20 inspected datasets. A complete run involves 1.2 million comparisons; a temporary disk index bounds memory use, and only the final CSV and JSON summaries are kept. No agreement tolerance is imposed, and PPML and GPML estimates are not expected to agree with each other.

The repository's [`saved/solvers/`](../saved/solvers/) holds the compact summaries of the completed study, which report mode uses to rebuild the published displays. Appendix quantiles use the inverse empirical distribution function.

## Code

| File | Role |
|---|---|
| `run.py` | Runs the study, applies the time budgets and writes `fits.jsonl` |
| `config.json` | Designs, sizes, seed, methods, IOLS controls and the 20 inspected datasets |
| `execution_schedule.json` | Draws run under process time limits in the original study |
| `src/designs.py` | The 20 simulation designs |
| `src/checks.py` | Structural screens and the independent accuracy checker |
| `src/iols_fit.py`, `src/historical_iols/` | IOLS (NumPy and the interface to the C++ kernel) |
| `src/statsmodels_fit.py` | statsmodels Newton and IRLS |
| `src/library/` | scikit-learn and SciPy solver loops (vendored, with their notices) |
| `native/` | C++ source of the IOLS kernel and its CMake build |
| `build_native.py` | Builds the C++ module for the current interpreter |
| `summarize.py`, `agreement.py`, `numeric_summary.py` | Summaries and pairwise agreement from new results |
