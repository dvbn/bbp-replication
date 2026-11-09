# Three-way fixed-effect simulations

This folder reproduces the fixed-effect simulations of Section 5.3 of *Dealing with Logs and Zeros in Regression Models* (Table 2, Figure 1) and the two covariance comparisons of Supplement S5.2. The simulations use the three-way gravity design of [Weidner and Zylkin (2021)](https://doi.org/10.1016/j.jinteco.2021.103513), "Bias and Consistency in Three-Way Gravity Models", *Journal of International Economics* 132, 103513, abbreviated WZ below.

## What is estimated

**Model.** For $N$ origins $i$, $N$ destinations $j$ and $T$ periods $t$,

$$E(Y_{ijt}\mid X,\alpha,\gamma,\eta)=\exp(X_{ijt}\beta_0+\alpha_{it}+\gamma_{jt}+\eta_{ij}),\qquad \beta_0=1.$$

The data follow the design of WZ's Tables 1 to 3. The origin-time, destination-time and pair effects are normal with standard deviation 1/16. The regressor is $X_{ij1}=\alpha_{i1}+\gamma_{j1}+\nu_{ij1}$ and $X_{ijt}=X_{ij,t-1}/2+\alpha_{it}+\gamma_{jt}+\nu_{ijt}$, with $\nu\sim N(0,1/4)$; the pair effect does not enter the regressor. The outcome is $Y_{ijt}=\lambda_{ijt}U_{ijt}$, where $\lambda_{ijt}$ is the conditional mean and $U_{ijt}$ is a mean-one lognormal error whose underlying normal follows an AR(1) process with coefficient 0.3 within each pair. The four DGPs choose the error variance so that the conditional variance of $Y$ is

| DGP | Conditional variance of $Y$ |
|---|---|
| 1 | $1$ (constant) |
| 2 | $\lambda$ (proportional to the mean) |
| 3 | $\lambda^2$ (proportional to the squared mean) |
| 4 | $0.5\,\lambda+0.5\,e^{2X}\lambda^2$ (depends on the regressor) |

All $N^2$ pairs are kept, including the diagonal pairs $i=j$.

**Estimators.** PPML and GPML are the estimating equations with working variance $\mu^\tau$ for $\tau=1$ and $\tau=2$. The paper's Table 2 compares:

- **GJ**: joint GPML, which estimates all three effect blocks.
- **GC**: GPML after cancelling the pair effects (Section 5.1 of the paper). Within each pair, the cancelled equation weights the ratios of the outcome to its fitted value by regressors centered within the pair, which removes the pair effect without estimating it; the origin-time and destination-time effects are still estimated.
- **GC+BC**: GC followed by the paper's estimating-equation correction of the remaining coefficient bias (Section 5.2), which adapts the empirical bias reduction of Kosmidis and Lunardon (2024) to the cancelled equation.
- **PJ**: PPML. At $\tau=1$, joint and pair-cancelled PPML give identical coefficients.
- **PJ+BC**: PPML with WZ's analytical coefficient correction.

Bias-corrected joint GPML (correction without cancellation) and jackknife estimates are not part of the experiment. GJ uses the sandwich of its own equation. All other estimators use the pair-clustered sandwich, evaluated at the uncorrected estimate for the corrected estimators. Coverage is the share of intervals $\hat\beta\pm1.96\,\widehat{\mathrm{SE}}$ that contain 1.

**Supplement S5.2.** A first table keeps the WZ-corrected PPML coefficient and replaces the ordinary sandwich with WZ's covariance adjustment, using all 5,000 replications at $T=2$. A separate pilot, on the first 100 base replications per DGP at $N=50$, $T=2$, holds the paper's generic bias-corrected coefficient fixed for cancelled PPML and cancelled GPML and compares the ordinary sandwich with the paper's higher-order adjustment, and, for PPML only, with WZ's covariance. The pilot is exploratory; negative adjusted variances are reported as invalid intervals rather than clipped.

## How the numbers are computed

Every fit solves the estimating equation with Newton's method using the actual Jacobian and Armijo backtracking on a fixed residual metric (at most 200 iterations and 60 backtracks). The metric and the common starting value come from a preliminary cancelled GPML fit (the "Gamma pilot"). A root is accepted only if a fresh evaluation of the original equation passes coordinate-wise residual and step tolerances ($10^{-10}$ and $10^{-9}$), the same tolerances in the fixed metric, and a condition-number check; standard errors are reported only if the remaining numerical correction is below $10^{-6}$ standard errors. When a direct joint fit fails, a bounded continuation in $\tau$ from a certified neighboring root provides an additional start. The tolerances are listed in [`config.json`](config.json).

The WZ coefficient and covariance corrections follow the formulas of WZ (Section 3.4 and Supplement A.2, A.3 and A.7), adapted to panels that include the diagonal pairs. This is an independent implementation, not the authors' Stata command; exact parity with that command on diagonal-including panels has not been established. The paper also reports discrepancies with separate Stata fitted-value and reference-score checks, which preclude describing these simulations as an exact reproduction of WZ's published numbers.

## Run

From the repository root, after installing `requirements.txt` (NumPy and SciPy suffice):

```bash
python wz/run.py --mode smoke --output outputs/wz-smoke   # DGP 1, N=20, T=2, replication 1, plus one pilot case
python wz/run.py --mode full --output outputs/wz          # complete experiment and pilot
```

The smoke run checks that the code executes, not Monte Carlo precision. The complete experiment fits all four DGPs in the nine base cells ($N\in\{20,50,100\}$, $T\in\{2,5,10\}$) and the $N=500$, $T=2$ extension, each with 5,000 replications, and then runs the 100-replication pilot. This is a large computation.

| Option | Effect |
|---|---|
| `--paper-only` | Estimate only the $T=2$ cells used in the paper; seeds and master dimensions are unchanged |
| `--replications R` | Use the original replications 1 to R of each cell; unless `--pilot-replications` is given, the pilot uses the first min(R, 100) |
| `--dgps 1,2` | Select DGPs |
| `--cells 20x2,50x2` | Select $N\times T$ cells |
| `--workers 2` | Number of worker processes |
| `--skip-pilot`, `--pilot-only` | Omit the covariance pilot, or run only the pilot |
| `--pilot-replications R` | Use the first R pilot replications (at most 100) |
| `--resume` | Continue an interrupted run in the same output directory |

For example, a small check of the paper cells without the pilot:

```bash
python wz/run.py --mode full --replications 2 --paper-only --skip-pilot --output outputs/wz-small
```

Selected DGPs, cells or replication counts change what the run covers and must be stated when reporting results. Numerical runs use one BLAS thread per worker and require Linux or macOS. A different configuration, source version or NumPy/SciPy version requires a fresh output directory, and an output directory accepts one writer at a time. If a run is interrupted while writing, the incomplete final record is reported for recovery rather than discarded silently.

## Random numbers and replications

Each base replication draws one $100\times100\times10$ master array from seed 20261002, and each cell uses its first $N$ origins, first $N$ destinations and first $T$ periods. Cells and DGPs of a replication therefore share their draws, which makes comparisons across $N$, $T$ and DGPs paired. The $N=500$ extension uses an independent seed (20261004) and a $500\times500\times2$ master, so it is not paired with the base cells. The generator is the original Philox counter-based stream keyed by seed and replication, with a raw-word Box-Muller transform, so the draws do not depend on NumPy's normal-sampling algorithm or on the processing order. Monte Carlo standard errors of the summaries use a bootstrap over replications with 999 resamples and seed 20261003.

## Source versions

The base experiment was computed in two stages, and both implementations are included:

- [`src/original/`](src/original/) produced the 33,979 base cases that are kept from the first stage: DGP 1 at $N=20$ and $N=50$ for every $T$, and DGP 1 at $N=100$, $T=2$ for replications 1 to 3,979.
- [`src/repaired/`](src/repaired/) produced all other base cases, the $N=500$ extension and the pilot. It refines the admission of the Gamma pilot fit as the common starting value; the estimating equations and tolerances are unchanged.

The runner selects the version for each case as in the completed study (`source_selection` in [`config.json`](config.json)). It does not replace the original successful estimates with a rerun of the repaired version. Apart from package configuration, the numerical code of both versions is unchanged from the research implementation.

## Code

| File | Role |
|---|---|
| `run.py` | Runs the experiment and the pilot, selects the source version, writes summaries |
| `config.json` | Designs, seeds, solver settings, tolerances and method list |
| `pilot_roots.py` | Cancelled roots and scalar derivatives for the pilot |
| `analytic_covariance.py` | Experimental higher-order covariance used only in the pilot |
| `src/*/wz_fe_sim/dgp/` | Random-number stream (`innovations.py`) and the WZ design (`transform.py`) |
| `src/*/wz_fe_sim/design/` | Panel cells and the pair-level evaluation of the cancelled and profiled joint equations |
| `src/*/wz_fe_sim/estimators/newton.py` | Newton solver and root certificate |
| `src/*/wz_fe_sim/estimators/factorial.py` | All fits of one replication and cell: Gamma pilot, cancelled and joint roots |
| `src/*/wz_fe_sim/estimators/correction.py` | Pair sandwich and the paper's coefficient correction |
| `src/*/wz_fe_sim/estimators/wz_tau1.py` | WZ coefficient correction and covariance adjustment for PPML |
| `src/*/wz_fe_sim/estimators/continuation.py` | Bounded $\tau$ continuation for failed joint fits |
| `src/*/wz_fe_sim/summarise/tables.py` | Cell summaries and bootstrap standard errors |

## Outputs

The output folder contains `base/`, `n500/`, `pilot/` and `settings.json`, which records the scientific settings and the installed dependency versions.

- `base/` and `n500/`: `summary.csv` (cell statistics), `methods.json` (method definitions) and one NPZ file per cell with arrays `<method>_beta` and `<method>_se`. Replication-level JSONL records keep root, correction and inference statuses and diagnostics; failed fits stay in the denominators.
- `pilot/`: `draw_results.csv`, `summary.csv` and `summary.json`. Roots are saved before the covariance calculations, so a covariance failure cannot erase them; such a failure stops the pilot and is recorded in `failures.jsonl`.

Before claiming a complete replication, check the counts of accepted fits and exclusions in these files. The repository's [`saved/wz/`](../saved/wz/) holds the paper's results for the $T=2$ cells, which report mode uses to rebuild the displays.
