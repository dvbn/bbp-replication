# Dealing with Logs and Zeros in Regression Models: replication files

This repository reproduces the numerical results of

> David Benatia, Christophe Bellégo and Louis-Daniel Pape, *Dealing with Logs and Zeros in Regression Models*. [arXiv:2203.11820](https://arxiv.org/abs/2203.11820).

The paper studies how to estimate proportional (percentage) effects when the outcome can be zero. It defines the target parameters through power means and weighted individual percentage changes, relates them to pseudo-maximum likelihood (PML) estimation of exponential conditional-mean models, introduces iterated ordinary least squares (IOLS), an algorithm that computes these estimators through repeated OLS regressions, and develops fixed-effect cancellation and bias correction for general working variances.

The repository covers the paper's three numerical studies:

| Study | Question | Paper | Folder |
|---|---|---|---|
| [Solver comparison](#1-solver-comparison) | How reliably and how fast do IOLS and four established solvers compute the same PPML and GPML estimates? | Section 4.3, Table 1; Supplement S5.1 | [`solvers/`](solvers/README.md) |
| [Three-way fixed-effect simulations](#2-three-way-fixed-effect-simulations) | How do joint estimation, cancellation and bias correction perform in the gravity panel design of Weidner and Zylkin (2021)? | Section 5.3, Table 2, Figure 1; Supplement S5.2 | [`wz/`](wz/README.md) |
| [Empirical application](#3-empirical-application) | Does the estimated association between pre-colonial political centralization and nighttime light density depend on the transformation, the working variance and the fixed-effect treatment? | Section 6, Table 3; Supplement S6 | [`empirical/`](empirical/README.md) |

Section, table and figure numbers refer to the October 2026 manuscript revision and can differ from arXiv v3 (September 2025). The numerical routines are those that produced the paper's results; the runners and reporting scripts were written for this repository. It is independent of the general-purpose `powermean` package, which is developed separately and is not needed here.

## Quick start: rebuild the paper's tables and figures

The paper's displays can be rebuilt from the saved estimates in a few seconds, without estimation, compilation, a LaTeX installation or an internet connection:

```bash
python3.14 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py --mode report --check-reference
```

The command writes 13 LaTeX table files, CSV exports of most of their numbers, and the two figures in PDF and PNG format to `outputs/report/paper/`. With `--check-reference`, it also compares the numerical rows of the generated tables with the paper's tables in [`reference/`](reference/). Only NumPy and Matplotlib are needed for this step. The checked environment is CPython 3.14.3 with the versions pinned in [`requirements.txt`](requirements.txt). The [replication guide](docs/replication.md#paper-displays-and-generated-files) maps each generated file to its paper display.

To rerun the estimation itself, see [Rerunning the computations](#rerunning-the-computations).

## 1. Solver comparison

*Folder [`solvers/`](solvers/README.md). Paper: Section 4.3 and Table 1; Supplement S5.1.*

**Model and estimators.** The simulated data follow an exponential conditional mean without fixed effects, $E(Y\mid X)=\exp(X'\beta)$, where $X$ contains an intercept, a binary treatment with coefficient $\log(1.08)$ (an 8 percent effect on the mean) and nuisance regressors. Each dataset is fitted by PPML and by GPML, which solve

$$\frac{1}{n}\sum_{i=1}^n X_i\,\mu_i(\beta)^{1-\tau}\bigl(Y_i-\mu_i(\beta)\bigr)=0,\qquad \mu_i(\beta)=\exp(X_i'\beta),$$

with working-variance power $\tau=1$ (PPML) or $\tau=2$ (GPML). Both equations admit zero outcomes. Six methods solve the same two equations: IOLS (the paper's algorithm, in C++ and in NumPy), statsmodels' Newton method (observed Hessian) and IRLS (Fisher scoring), scikit-learn's Newton–Cholesky solver with backtracking line search, and SciPy's L-BFGS-B.

**Design.** There are 16 ordinary designs and four stress designs (high dispersion, heavy tails, rare large outcomes and near collinearity), each at $(n,K)=(1{,}000,10)$ and $(10{,}000,100)$, with 1,000 draws per design and size: 40,000 datasets and 480,000 scheduled method-equation records. A fit counts as successful only if its output passes a common external accuracy check at $10^{-8}$ within 120 CPU seconds and 300 wall-clock seconds, whatever the solver itself reports.

**Main result (Table 1).** Each cell reports the **failure rate (%) · median/mean CPU time (ms)**. Times include every attempted fit, with each recorded CPU time capped at 120 seconds; setup time is excluded. Certified structural exclusions (datasets for which the equation has no unique finite solution) are not attempted fits and are left out of both summaries.

| Designs | n | Equation | IOLS (C++) | Newton | IRLS |
|---|---:|---|---:|---:|---:|
| Ordinary | 1,000 | PPML | 0.00 · 2.4/2.7 | 0.00 · 1.0/1.1 | 0.00 · 1.2/1.3 |
| Ordinary | 1,000 | GPML | 0.00 · 2.7/8.1 | 1.94 · 1.0/28 | 0.07 · 2.8/12 |
| Ordinary | 10,000 | PPML | 0.00 · 106/111 | 0.00 · 67/70 | 0.00 · 200/206 |
| Ordinary | 10,000 | GPML | 0.00 · 115/118 | 0.32 · 67/102 | 0.04 · 555/722 |
| Stress | 1,000 | PPML | 0.00 · 3.4/4.8 | 0.15 · 1.2/9.3 | 0.15 · 1.5/7.8 |
| Stress | 1,000 | GPML | 0.00 · 3.3/6.5 | 14.62 · 1.5/436 | 40.98 · 3.4/2,223 |
| Stress | 10,000 | PPML | 0.00 · 158/202 | 4.67 · 113/1,643 | 0.35 · 397/496 |
| Stress | 10,000 | GPML | 0.00 · 144/192 | 27.75 · 105/14,831 | 75.02 · 339/419 |

C++ IOLS passes every attempted $10^{-8}$ accuracy check within the budget. Newton and IRLS are the statsmodels implementations. The [full table](reference/tables/solver_preliminary_main.tex) and its [numerical inputs](saved/solvers/paper_main_summary.csv) are included.

![Percent of unsuccessful PPML and GPML fits for IOLS, Newton, IRLS, damped Newton and L-BFGS-B at the strict and secondary accuracy thresholds.](docs/figures/solver_stress_accuracy.png)

*Supplement figure: unsuccessful fits in the larger stress designs ($n=10{,}000$, $K=100$), 4,000 attempted fits per method and equation. Blue bars use the $10^{-8}$ accuracy check and gray bars the secondary $10^{-6}$ check, with the same computing budget. NumPy IOLS results are reported separately in the Supplement tables.*

## 2. Three-way fixed-effect simulations

*Folder [`wz/`](wz/README.md). Paper: Section 5.3, Table 2 and Figure 1; Supplement S5.2.*

**Model.** The data follow the three-way gravity design of [Weidner and Zylkin (2021)](https://doi.org/10.1016/j.jinteco.2021.103513):

$$E(Y_{ijt}\mid X,\alpha,\gamma,\eta)=\exp(X_{ijt}\beta_0+\alpha_{it}+\gamma_{jt}+\eta_{ij}),\qquad \beta_0=1,$$

with origin-time effects $\alpha_{it}$, destination-time effects $\gamma_{jt}$ and pair effects $\eta_{ij}$, for $N$ origins, $N$ destinations and $T$ periods; all $N^2$ pairs are kept, including $i=j$. The four data-generating processes (DGPs) share this mean and differ in the conditional variance of $Y$: constant (DGP 1), proportional to the mean (DGP 2), proportional to the squared mean (DGP 3), and a regressor-dependent mixture (DGP 4). With $T$ fixed and $N$ growing, each pair effect is estimated from only $T$ observations, which creates an incidental-parameter problem. Cancelling the pair effects removes them from the estimating equation, but the origin-time and destination-time effects that remain can still bias the coefficient by an amount comparable to its standard error; the bias correction addresses this second problem.

**Estimators.** The table and figure compare five procedures:

| Label | Estimator | Standard error |
|---|---|---|
| GJ | Joint GPML: all three effect blocks estimated | Sandwich of the joint GPML equation |
| GC | GPML after cancelling the pair effects; origin-time and destination-time effects are estimated | Pair-clustered sandwich |
| GC+BC | GC followed by the paper's estimating-equation bias correction | Pair-clustered sandwich at the uncorrected estimate |
| PJ | PPML; joint and pair-cancelled PPML give identical coefficients | Pair-clustered sandwich |
| PJ+BC | PPML with the analytical coefficient correction of Weidner and Zylkin (2021) | Pair-clustered sandwich at the uncorrected estimate |

**Design.** The full experiment has 5,000 replications for every DGP and every $N\in\{20,50,100\}$, $T\in\{2,5,10\}$, and an independent extension with 5,000 replications per DGP at $N=500$, $T=2$. The paper's displays use $T=2$.

**Main result (Table 2).** Each DGP cell reports **bias / mean SE / coverage (%)**, where bias and mean SE are in units of the empirical standard deviation of the estimates and coverage is that of nominal 95% normal intervals; 5,000 replications per cell, $T=2$.

| N | Method | DGP 1 | DGP 2 | DGP 3 | DGP 4 |
|---:|---|---:|---:|---:|---:|
| 50 | GJ | 11.48 / 0.98 / 0.0 | 6.52 / 0.99 / 0.0 | -0.02 / 0.98 / 94.3 | -3.59 / 0.96 / 4.7 |
| 50 | GC | 0.63 / 0.88 / 84.3 | 0.36 / 0.91 / 89.8 | -0.02 / 0.90 / 91.8 | -0.57 / 0.83 / 83.3 |
| 50 | GC+BC | 0.20 / 0.85 / 89.1 | 0.09 / 0.88 / 91.4 | -0.02 / 0.86 / 90.9 | -0.34 / 0.77 / 84.5 |
| 50 | PJ | 0.58 / 0.89 / 85.5 | 0.35 / 0.92 / 90.3 | 0.01 / 0.90 / 92.0 | -0.55 / 0.83 / 84.3 |
| 50 | PJ+BC | 0.10 / 0.85 / 89.8 | 0.04 / 0.89 / 91.7 | -0.03 / 0.86 / 90.8 | -0.30 / 0.77 / 84.6 |
| 100 | GJ | 22.64 / 0.96 / 0.0 | 13.16 / 0.99 / 0.0 | -0.02 / 0.99 / 94.5 | -7.19 / 0.97 / 0.0 |
| 100 | GC | 0.66 / 0.94 / 86.7 | 0.36 / 0.96 / 91.9 | -0.03 / 0.94 / 93.2 | -0.65 / 0.85 / 82.3 |
| 100 | GC+BC | 0.12 / 0.91 / 92.1 | 0.04 / 0.94 / 93.0 | -0.02 / 0.91 / 92.2 | -0.33 / 0.79 / 85.5 |
| 100 | PJ | 0.59 / 0.95 / 88.8 | 0.35 / 0.96 / 92.5 | 0.01 / 0.94 / 93.4 | -0.64 / 0.85 / 83.2 |
| 100 | PJ+BC | 0.04 / 0.92 / 92.6 | 0.00 / 0.95 / 93.6 | -0.03 / 0.91 / 92.4 | -0.29 / 0.79 / 86.0 |
| 500 | GJ | 99.63 / 0.84 / 0.0 | 64.93 / 0.97 / 0.0 | -0.01 / 0.99 / 95.4 | -32.91 / 0.89 / 0.0 |
| 500 | GC | 0.70 / 0.97 / 87.7 | 0.39 / 0.98 / 92.5 | -0.02 / 0.97 / 94.4 | -0.84 / 0.92 / 81.7 |
| 500 | GC+BC | 0.05 / 0.95 / 93.7 | 0.01 / 0.97 / 94.4 | -0.02 / 0.97 / 94.3 | -0.26 / 0.87 / 89.9 |
| 500 | PJ | 0.59 / 0.98 / 89.5 | 0.36 / 0.98 / 92.9 | 0.01 / 0.98 / 94.6 | -0.85 / 0.92 / 81.9 |
| 500 | PJ+BC | 0.00 / 0.96 / 94.0 | -0.01 / 0.98 / 94.6 | -0.02 / 0.97 / 94.3 | -0.23 / 0.87 / 90.3 |

Joint GPML concentrates away from the true coefficient in DGPs 1, 2 and 4. Cancellation substantially reduces the GPML bias, and the coefficient correction delivers performance comparable to WZ-corrected PPML. Undercoverage remains, most pronounced in DGP 4, and the comparison is less favorable with more periods: for DGP 1 at $N=100$, corrected GPML coverage is 81.5% at $T=5$ and 57.9% at $T=10$ (Supplement S5.2). The $N=500$ draws are independent of the smaller-$N$ draws. See the [full table](reference/tables/fe_T2_main_body.tex); the Supplement tables compare the ordinary pair sandwich with the covariance adjustment of Weidner and Zylkin (2021) and report a separate higher-order covariance pilot.

![GPML coefficient densities across four DGPs for joint fixed effects, cancellation and cancellation with correction, at N=20, 50 and 100.](docs/figures/fe_gpml_T2_compact.png)

*Figure 1: kernel densities of the GPML estimates at $T=2$ and $N=20, 50, 100$, using all 5,000 estimates per curve. The dotted line marks the true coefficient, $\beta_0=1$.*

## 3. Empirical application

*Folder [`empirical/`](empirical/README.md). Paper: Section 6 and Table 3; Supplement S6.*

**Question and data.** The application revisits the nighttime-light regression of [Michalopoulos and Papaioannou (2013)](https://doi.org/10.3982/ECTA9613), Table III, column 4. The unit of observation is an ethnic homeland within a country. The outcome $Y$ is nighttime light density, which is zero for 164 of the 682 observations in the full sample. The regressor of interest is a jurisdictional hierarchy index (0 to 4) measuring pre-colonial political centralization. All specifications keep the original eleven controls and country effects, and cluster standard errors by country and by ethnic family.

**Specifications.** For each sample, the application estimates:

- PPML and GPML of $Y$ with country effects, estimated jointly (main specification) or cancelled (sensitivity check). The reported quantity, $100(\exp(\hat\beta)-1)$, is the percentage difference in mean light density per hierarchy category.
- The same equations for $\sqrt{Y}$ (the half-power mean, Supplement S6).
- OLS of $\log(Y+c)$ with absorbed country effects for $c=10^{-5}$, $0.01$ (the original specification) and $1$. Here $100(\exp(\hat b)-1)$ refers to the geometric mean of $Y+c$, not to mean light density.

Because the brightest homeland accounts for about 10 percent of total light density, the estimates are also computed after removing the brightest observation, the brightest 1, 2 and 5 percent, and all zeros. These outcome-based restrictions are sensitivity checks, not preferred samples.

**Main result (Table 3).** Each cell reports the **percentage association (two-sided p-value)**. Standard errors use the signed two-way cluster covariance (country plus ethnic family minus their intersection), without a finite-sample factor.

| Specification | Full (n=682) | Drop 2% (n=668) | Drop 5% (n=646) | Positive only (n=517) |
|---|---:|---:|---:|---:|
| OLS, c=0.00001 | 19.90 (0.122) | 25.29 (0.066) | 19.93 (0.196) | 16.12 (0.014) |
| OLS, c=0.01 | 19.32 (<0.001) | 20.49 (<0.001) | 16.20 (0.003) | 17.18 (<0.001) |
| OLS, c=1 | 5.44 (<0.001) | 4.86 (<0.001) | 2.85 (0.001) | 4.99 (0.011) |
| PPML (joint = cancelled) | -10.46 (0.226) | 21.43 (0.040) | 8.24 (0.052) | -8.57 (0.360) |
| GPML (joint) | 16.04 (0.011) | 19.80 (0.002) | 15.95 (0.032) | 17.27 (0.005) |
| GPML (cancelled) | 14.88 (0.019) | 34.93 (<0.001) | 15.94 (0.002) | 11.56 (0.080) |

These are conditional associations. The [full table](reference/tables/application_sensitivity.tex) and the Supplement tables report the remaining samples, confidence intervals, half-power models and all coefficients.

**Data.** The bundled input is an attributed subset of the replication data of Michalopoulos and Papaioannou (2013), redistributed under the publisher's terms. [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) documents the source, the preparation and how to rebuild the file from the original archive.

## Rerunning the computations

Report mode reuses the saved estimates. The two other modes estimate the models again.

**Small example (`--mode smoke`).** Runs one WZ draw and one covariance-pilot case, both equations through all six solvers, and the empirical full-sample specifications. Results go to `outputs/smoke/`. The C++ IOLS solver must be compiled first; this requires CMake, a C++17 compiler, BLAS/LAPACK and the [Eigen](https://eigen.tuxfamily.org/) headers (version 3.4 or later; the checked build used Eigen 5.0.1):

```bash
python solvers/build_native.py --eigen-include /path/to/eigen3   # directory containing Eigen/Core
python run.py --mode smoke
```

The build script uses the current Python environment, an optimized release configuration and the historical floating-point flags; it downloads nothing, and no compiled binary is distributed. Without it, the solver study stops before fitting and asks for the build, so `python run.py --mode smoke` ends after the WZ study and never reaches the application. The WZ and empirical studies do not need the module and can be run separately:

```bash
python run.py --mode smoke --study wz
python run.py --mode smoke --study empirical
```

**Complete rerun (`--mode full`).** Regenerates every simulated dataset from the original seeds, refits all models and rebuilds the displays from the new results in `outputs/full/paper/`:

```bash
python run.py --mode full
```

This is a large computation: 180,000 WZ draw-cells in the base experiment, 20,000 in the $N=500$ extension, the 400-case covariance pilot, 480,000 scheduled solver records on 40,000 datasets, and the six empirical samples. By default, execution is sequential, with one BLAS/OpenMP thread per process. The simulation runners require Linux or macOS. Completed records are saved as they finish; to continue an interrupted run, repeat the same command with `--resume`. A different configuration needs a fresh output directory. Restricted runs:

```bash
python run.py --mode full --study wz --paper-only     # WZ cells with T = 2 only
python run.py --mode full --study solvers
python run.py --mode full --study empirical           # needs neither the C++ module nor internet access
python run.py --mode full --replications 2 --output outputs/small
```

`--paper-only` keeps the original seeds and master dimensions, so the $T=2$ cells use the same draws as in the complete experiment. `--replications R` keeps the first R original draws of every cell (and at most R replications of the WZ covariance pilot); it produces a numerical subset and skips automatic reporting, since a subset is not the Monte Carlo result. Results produced in another directory can be turned into the paper displays with

```bash
python run.py --mode report --input-root outputs/full --output outputs/rebuilt
```

Reruns retain the reported designs, estimating equations and acceptance rules. The [application README](empirical/README.md#computation) notes differences from Supplement S6's broader description of computation. Failed fits, structural exclusions and invalid intervals are kept in the results. Failed fits count in the reported denominators, while certified structural exclusions of the solver study are reported separately, as in the paper; no estimator is substituted, no threshold is relaxed, and C++ IOLS is never silently replaced by the NumPy version. New CPU timings depend on the machine and libraries, whereas report mode reproduces the paper's original timings. The study READMEs list the options for selecting designs, cells and methods.

## Repository layout

| Path | Contents |
|---|---|
| `run.py` | Single entry point for the three modes and three studies |
| `solvers/` | Solver comparison: simulation designs, IOLS and comparator implementations, C++ sources and build script ([README](solvers/README.md)) |
| `wz/` | Three-way fixed-effect simulations and covariance pilot ([README](wz/README.md)) |
| `empirical/` | Application data, sample rules, estimation and inference ([README](empirical/README.md)) |
| `reporting/` | Scripts that turn saved or new results into the paper's tables and figures |
| `saved/` | Numerical results used by report mode |
| `reference/` | Paper tables for numerical checks and original figures for visual comparison |
| `docs/` | [Replication guide](docs/replication.md) and README figures |

## Citation

Please cite the paper when using these files:

```bibtex
@misc{benatia2022logs,
  title         = {Dealing with Logs and Zeros in Regression Models},
  author        = {Benatia, David and Bell{\'e}go, Christophe and Pape, Louis-Daniel},
  year          = {2022},
  eprint        = {2203.11820},
  archivePrefix = {arXiv},
  primaryClass  = {econ.EM},
  url           = {https://arxiv.org/abs/2203.11820}
}
```

Work using the three-way fixed-effect simulations should also cite the study whose design and PPML corrections they use:

```bibtex
@article{weidner2021bias,
  title   = {Bias and Consistency in Three-Way Gravity Models},
  author  = {Weidner, Martin and Zylkin, Thomas},
  journal = {Journal of International Economics},
  volume  = {132},
  pages   = {103513},
  year    = {2021},
  doi     = {10.1016/j.jinteco.2021.103513}
}
```

Work using the application data should cite the original article and its replication package (see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)):

```bibtex
@article{michalopoulos2013precolonial,
  title   = {Pre-Colonial Ethnic Institutions and Contemporary African Development},
  author  = {Michalopoulos, Stelios and Papaioannou, Elias},
  journal = {Econometrica},
  volume  = {81},
  number  = {1},
  pages   = {113--152},
  year    = {2013},
  doi     = {10.3982/ECTA9613}
}
```

Citation metadata are also in [`CITATION.cff`](CITATION.cff).

## License

The code written for this repository is released under the [BSD-3-Clause license](LICENSE). Bundled third-party code keeps its original notices, reproduced in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The application data remain subject to their source's terms and are not covered by the code license.
