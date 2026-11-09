# Replication guide

This guide explains how each result of *Dealing with Logs and Zeros in Regression Models* (Benatia, Bellégo and Pape) is produced from the code in this repository: which file each display comes from, how the simulations are generated, which estimators and acceptance rules they use, and what a rerun can and cannot reproduce. The [README](../README.md) gives the quick start; the study READMEs ([`solvers/`](../solvers/README.md), [`wz/`](../wz/README.md), [`empirical/`](../empirical/README.md)) give the commands and options of each study.

Section, table and figure numbers refer to the October 2026 manuscript revision.

## Two ways to reproduce the results

- **Report mode** (`python run.py --mode report`) rebuilds the tables and figures from the saved estimates in `saved/`. It fits no model, so it reproduces the paper's numbers exactly, including the original CPU timings. With `--check-reference`, it compares the printed values with the paper's tables in `reference/`.
- **Numerical runs** (`--mode smoke` and `--mode full`) regenerate the simulated data from the original seeds, or read the application data, refit every model and save new results. A full run then builds the displays from these new results only; it never falls back on the saved summaries. New timings depend on the hardware and libraries, and rare numerical boundary cases can change with the BLAS library, compiler or floating-point environment. A smoke run or a subset of replications shows that the code runs; it does not replace the complete experiments.

## Paper displays and generated files

All generated files are written to the `paper/` subdirectory of the chosen output folder. Most LaTeX tables have a CSV export of their numbers (the safeguarded-solver table and the all-coefficient tables do not), and each figure is accompanied by the numerical data it plots.

| Generated file | Paper display | Contents |
|---|---|---|
| `solver_preliminary_main.tex` | Table 1 (Section 4.3) | Failure rates and capped median/mean CPU times of C++ IOLS, statsmodels Newton and statsmodels IRLS |
| `solver_preliminary_appendix.tex` | Supplement S5.1 | Outcomes and raw CPU times of NumPy IOLS, C++ IOLS, Newton and IRLS, ordinary and stress designs |
| `solver_safeguarded_appendix.tex` | Supplement S5.1 | Outcomes of scikit-learn's Newton–Cholesky solver and SciPy's L-BFGS-B; failures of all six methods under the common budget |
| `solver_statistical_stress.tex` | Supplement S5.1 | Bias and dispersion of the C++ IOLS estimates in the stress designs: on each equation's accurate fits, and on the draws where both statsmodels solvers fail (moments conditional on that selection) |
| `solver_stress_accuracy.pdf` | Supplement S5.1 figure | Unsuccessful fits in the larger stress designs for five methods, two equations and two accuracy thresholds |
| `solver_agreement.csv`, `solver_agreement_counts.json` | Supplement S5.1 text | Pairwise agreement of coefficients and fitted indices across methods |
| `fe_T2_display.tex`, `fe_T2_main_body.tex` | Table 2 (Section 5.3) | Bias, standard errors and coverage at $N=50,100,500$, $T=2$ |
| `fe_gpml_T2_compact.pdf` | Figure 1 (Section 5.3) | Densities of the GPML estimates at $N=20,50,100$, $T=2$ |
| `fe_wz_covariance_comparison_T2.tex` | Supplement S5.2 | Ordinary pair sandwich versus the covariance adjustment of Weidner and Zylkin (2021), at the same WZ-corrected PPML coefficient |
| `fe_higher_order_covariance_pilot_T2.tex` | Supplement S5.2 | Exploratory covariance comparison on the first 100 draws at $N=50$, $T=2$ |
| `application_sensitivity.tex` | Table 3 (Section 6) | Percentage associations and p-values in four samples |
| `application_full_grid.tex`, `application_full_intervals.tex` | Supplement S6 | Arithmetic- and half-power-mean and shifted-log results in five samples: associations, p-values and 95% intervals |
| `application_positive_only.tex` | Supplement S6 | Positive-outcome-only sample, all nine distinct models |
| `application_all_coefficients.tex` | Supplement S6 | Nine tables of all slope coefficients and standard errors |

The five application files contain 13 tables in total.

The FE figure has 36 curves (four DGPs, three GPML estimators and three sample sizes). Every draw contributes to its density; the plotting window does not trim the sample. Densities use the original Silverman bandwidth rule and display window. Because the final historical plotting script was not available, the renderer recreates the plotted numbers with its own typography and layout; the original PDFs are in `reference/figures/` for comparison.

`--check-reference` compares the data rows of 12 table files with the paper's tables: seven simulation files (the FE table is checked through `fe_T2_main_body.tex`) and the five application files. The comparison covers the printed coefficients, standard errors, significance stars, p-values and confidence intervals. Panel header rows, such as the attempt and exclusion counts printed above the panels of the solver appendix tables, are not compared.

## Solver comparison

**Purpose.** Section 4.3 asks whether IOLS computes PPML and GPML estimates reliably and at competitive cost compared with standard solvers applied to the same estimating equations.

**Designs.** All designs satisfy $E(Y\mid X)=\exp(X'\beta_0)$, with an independent Bernoulli(1/2) treatment whose coefficient is $\log(1.08)$ and an untreated population mean of 2. Sixteen ordinary designs vary the outcome distribution (Poisson; Gamma; negative binomial with three dispersion levels; compound Poisson-Gamma; inverse Gaussian; heterogeneous negative binomial; lognormal; zero-inflated lognormal) and the regressor distributions. Four stress designs add high dispersion, heavy tails, rare large outcomes through contamination, or near collinearity. Each design is drawn 1,000 times at $(n,K)=(1{,}000,10)$ and at $(10{,}000,100)$, where $K$ counts the intercept and the treatment: 40,000 datasets. The master seed is 2026100301, and the original independent streams for regressors, outcomes and method order are kept. Supplement S5.1 describes every design.

**Methods and equations.** Each dataset is fitted by PPML ($\tau=1$) and by GPML ($\tau=2$, the log-link relative-residual equation, including zero outcomes) with six methods: IOLS in NumPy and in C++, statsmodels' Newton and IRLS, scikit-learn's Newton–Cholesky solver with backtracking line search, and SciPy's L-BFGS-B, giving 480,000 scheduled records. IOLS starts from its internal OLS fit of $\log(1+y)$; the other methods start at zero slopes with the intercept at $\log(\bar y)$. The comparator routines are the historical ones, run with pinned library versions.

**Structural exclusions.** Before fitting, and outside the timed fits, a screen classifies each dataset and equation: a rank-deficient design has no unique solution, and a linear-programming certificate can show that no finite solution exists. A case certified in this way is recorded as a structural exclusion with a placeholder record; the dataset is never redrawn, and no design column is dropped. Cases with zeros but no certificate are attempted.

**Accuracy.** Native convergence flags are not trusted. Every returned coefficient vector is checked against the original estimating equation: in QR-standardized coordinates, the Euclidean norm of the score and the root-mean-square local correction to the fitted index must both be at most $10^{-8}$ (strict check) or $10^{-6}$ (secondary check), and the observed curvature must be numerically positive definite. A later numerical review of the completed study retained 200 index-diagnostic qualifications (198 score-rejected rows and two rows whose classification was verified in high precision although an index scalar was not reproduced); these did not change any acceptance flag. The public runner applies the original checker but does not include that private review.

**Time budget.** A fit succeeds under the common budget only if it passes the accuracy check within 120 CPU seconds and 300 wall-clock seconds. Table 1 reports median and mean CPU times over all attempted fits, with each recorded CPU time capped at 120 seconds; the Supplement tables report the raw times. Percentages are computed in the original order of operations (division before multiplication), so the printed values match exactly. In the original study, most calls ran with iteration and evaluation limits but without a process timer, and 53,536 completion calls ran under the 120/300-second limits. A full rerun reproduces this distinction by default (see the [solver README](../solvers/README.md)).

**Disclosed tuning.** The IOLS damping grid used in the reported runs was revised after inspecting 20 datasets from the reported sample. The paper reports this as a retrospective update, not as an untouched confirmation, and the agreement summaries are also computed without those 20 datasets. Their keys are listed in `solvers/config.json`.

## Three-way fixed-effect simulations

**Purpose.** Section 5.3 compares joint estimation, cancellation of the pair effects, and cancellation followed by bias correction in the gravity design of Weidner and Zylkin (2021).

**Design.** The mean is $E(Y_{ijt}\mid\cdot)=\exp(X_{ijt}\beta_0+\alpha_{it}+\gamma_{jt}+\eta_{ij})$ with $\beta_0=1$, following the design of Weidner and Zylkin's Tables 1 to 3: the effects are normal with standard deviation 1/16, the regressor depends on the origin-time and destination-time effects and on its own lag with coefficient 1/2, and the multiplicative lognormal errors follow an AR(1) process with coefficient 0.3 within each pair. The four DGPs set the conditional variance to a constant (DGP 1), to the mean (DGP 2), to the squared mean (DGP 3), or to a mixture depending on the regressor (DGP 4). All $N^2$ pairs are kept, including $i=j$.

**Replications and seeds.** The base experiment has four DGPs, $N\in\{20,50,100\}$, $T\in\{2,5,10\}$ and 5,000 replications per cell. Each replication draws one $100\times100\times10$ master array (seed 20261002), and every cell takes its first $N$ origins, first $N$ destinations and first $T$ periods, so cells and DGPs of the same replication share their draws. The $N=500$, $T=2$ extension uses an independent seed (20261004) and a $500\times500\times2$ master, so its replication 1 is not paired with replication 1 of the base. The random numbers come from the original Philox counter-based generator with a raw-word Box-Muller transform, which makes them independent of NumPy's normal-sampling algorithm. Monte Carlo standard errors of the summary statistics use a bootstrap with 999 resamples and seed 20261003.

**Estimators.** Results are saved under method identifiers that map to the paper's labels as follows:

| Paper label | Saved ID | Estimator | Standard error |
|---|---|---|---|
| GJ | `GJ` | Joint GPML, all three effect blocks estimated | Sandwich of the joint GPML equation |
| GC | `G` | GPML with pair effects cancelled | Pair-clustered sandwich |
| GC+BC | `GB` | GC with the paper's coefficient bias correction | Pair-clustered sandwich at the uncorrected root |
| PJ | `P` | PPML (joint and cancelled coefficients coincide) | Pair-clustered sandwich |
| PJ+BC | `PW` | PPML with the coefficient correction of Weidner and Zylkin (2021) | Pair-clustered sandwich at the uncorrected root |
| (Supplement) | `PWa` | Same coefficient as `PW` | Covariance adjustment of Weidner and Zylkin (2021) |

The saved files also contain the other combinations of PPML coefficient (`P`, `PB`, `PW`) and covariance (ordinary sandwich, WZ unadjusted, WZ adjusted) listed in `saved/wz/base/methods.json`. Jackknife estimates are disabled.

**Two implementations of the same equations.** The base experiment was computed in two stages. The 33,979 base cases that the original implementation solved successfully are kept: DGP 1 at $N=20$ and $N=50$ for every $T$, and DGP 1 at $N=100$, $T=2$ for replications 1 to 3,979. All other base cases, the $N=500$ extension and the covariance pilot use a repaired implementation that refines the admission of the Gamma pilot fit as the common starting value; the estimating equations and tolerances are unchanged. Both versions are included (`wz/src/original/` and `wz/src/repaired/`), and the runner selects the version for each case explicitly. It does not replace the original successful estimates with a rerun of the repaired version.

**Covariance comparisons.** Supplement S5.2 contains two separate comparisons. The first uses all 5,000 replications at $T=2$ and holds the WZ-corrected PPML coefficient fixed while switching between the ordinary pair sandwich (`PW`) and the WZ-adjusted covariance (`PWa`). The second is an exploratory pilot on the first 100 base replications per DGP at $N=50$, $T=2$: for cancelled PPML and cancelled GPML, it holds the paper's generic bias-corrected coefficient fixed and compares the ordinary sandwich with the paper's higher-order covariance adjustment, and, for PPML only, with the WZ covariance. Negative adjusted variances are reported as invalid intervals; they are not clipped.

## Empirical application

**Specification.** The application reproduces the nighttime-light regression of Michalopoulos and Papaioannou (2013), Table III, column 4: the outcome is nighttime light density, the regressor of interest is the jurisdictional hierarchy index (0 to 4), and the eleven original controls and country effects are kept. Joint estimation profiles the country effects; cancellation replaces each regressor with its deviation from the country mean. Both are computed for $Y$ and for $\sqrt{Y}$; half-power coefficients and standard errors are divided by $\omega=1/2$ before transformation. The three shifted-log specifications regress $\log(Y+c)$ on the same regressors with absorbed country effects. No incidental-parameter correction is applied.

**Samples.** The baseline keeps complete cases and removes the one country with a single observation (682 observations). Five restrictions start from the same baseline: removing the brightest observation (681), the brightest 1, 2 or 5 percent (675, 668, 646), or all zeros (517). Country singletons created by a restriction are removed again.

**Inference.** Standard errors use the signed two-way cluster covariance, the sum of the country and ethnic-family cluster terms minus the term of their intersections, built from observation-level influence contributions, without a finite-sample factor or an eigenvalue adjustment. Paired comparisons of PPML and GPML, or of half-power and arithmetic means, stack the influence contributions of both models on the same sample, so their covariance is taken into account.

**Outputs.** A complete run produces 792 slope rows and 2,002 country-intercept point estimates. Country-intercept standard errors were not exported by the original application; they are left empty rather than set to zero.

## Environment

The checked environment is macOS on Apple silicon with CPython 3.14.3, NumPy 2.4.2, SciPy 1.17.0, pandas 3.0.0, Matplotlib 3.10.8, scikit-learn 1.8.0 and statsmodels 0.15.0. The C++ module was built with pybind11 3.0.2 and Eigen 5.0.1, in an optimized release configuration with precise floating-point settings. Only this environment was checked. The simulation runners use POSIX process and timer facilities and therefore require Linux or macOS.
