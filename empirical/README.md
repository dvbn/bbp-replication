# Political centralization and nighttime lights

This folder reproduces the empirical application of Section 6 of *Dealing with Logs and Zeros in Regression Models* (Table 3) and the application tables of Supplement S6. It revisits the nighttime-light regression of [Michalopoulos and Papaioannou (2013)](https://doi.org/10.3982/ECTA9613), "Pre-Colonial Ethnic Institutions and Contemporary African Development", *Econometrica* 81(1), 113–152, Table III, column 4.

The application shows that the estimated association between pre-colonial political centralization and contemporary development can depend on three choices: the transformation of an outcome with zeros, the working variance of the PML estimator, and the treatment of the country fixed effects.

## Data

The unit of observation is an ethnic homeland within a country. The outcome $Y$ is nighttime light density, a proxy for economic development. The regressor of interest is the jurisdictional hierarchy index (`centr_tribe`), an ordinal measure of pre-colonial political centralization from 0 to 4 that enters linearly. The eleven original controls are log population density, distances to the capital, the sea coast and the national border, log water area, log land area, mean elevation, agricultural suitability, malaria suitability, and indicators for petroleum and diamonds. Every specification includes country effects.

The bundled file `data/greek_econometrica_data.csv` contains 683 observations from the original replication package, with only the variables used here, in the original row order. It deletes no observation; all sample rules are applied by the code. [DATA_AVAILABILITY.md](../DATA_AVAILABILITY.md) documents its source, preparation and terms of use. A fresh copy can be rebuilt from the publisher's archive:

```bash
python empirical/fetch_data.py --output outputs/data/recovered-data.csv
```

## Samples

All samples start from the same baseline: complete cases on the outcome, the twelve regressors and both cluster identifiers, after removing countries with a single observation. Each restriction is applied to this baseline, after which country singletons are removed again and the centering, column scaling, estimation and inference are recomputed.

| Sample | Observations |
|---|---:|
| Full | 682 |
| Brightest observation removed | 681 |
| Brightest 1% removed | 675 |
| Brightest 2% removed | 668 |
| Brightest 5% removed | 646 |
| Positive outcomes only | 517 |

The full sample has 164 zero outcomes, 48 countries and 94 ethnic families. Brightness ranks the raw outcome in descending order, with the original row position breaking ties; percentage cuts remove the ceiling of the stated share of all 682 baseline observations, zeros included. The 5 percent cut removes 35 observations and one resulting country singleton; the positive-only sample removes the 164 zeros and one resulting singleton. The brightest homeland accounts for about 10 percent of total light density, which motivates these checks. They are sensitivity analyses; none is a preferred sample.

## What is estimated

For each sample, the application estimates:

- **PPML and GPML of $Y$** ($\tau=1$ and $\tau=2$), with the country effects either estimated jointly, which is the main specification, or cancelled. Joint estimation profiles the country effects, which reduces the numerical dimension without changing the estimator. Cancellation replaces each regressor by its deviation from the country mean (Section 5.1 of the paper). At $\tau=1$, joint and cancelled PPML coincide; at $\tau=2$, cancelled GPML weights countries differently from joint GPML and is a separate estimator.
- **The same equations for $\sqrt{Y}$**, which estimate the conditional half-power mean $(E(Y^{1/2}\mid X))^2$ rather than the arithmetic mean. The fitted coefficient $\gamma$ of the model for $Y^{1/2}$ is converted to the power-mean coefficient $\gamma/\omega$ with $\omega=1/2$, and its influence contributions and standard errors are divided by $\omega$ in the same way.
- **OLS of $\log(Y+c)$** with absorbed country effects, for $c=10^{-5}$, $0.01$ (the original specification) and $1$.

Table 3 and the association tables of Supplement S6 report $100(\exp(b)-1)$ for the hierarchy coefficient $b$; the all-coefficient tables report the coefficients and standard errors of all twelve regressors. For PPML and GPML of $Y$, this is the percentage difference in arithmetic mean light density per hierarchy category; for models of $\sqrt{Y}$, it is the percentage difference in the half-power mean. For the shifted-log regressions, it refers to the geometric mean of $Y+c$, which is not a proportional difference in mean light density. These are conditional associations. No incidental-parameter correction is applied.

## Computation

The exponential equations are solved with the safeguarded Newton routines of the fixed-effect simulations, bundled in `vendor/wz_fe_sim` (version 0.2.0): Newton steps with the actual Jacobian and LU solves, and Armijo backtracking on a fixed positive-definite residual metric. A preliminary cancelled GPML fit (the Gamma pilot) defines the metric and the common starting value, and cancelled GPML is solved before cancelled PPML. When a prescribed start fails, the application also tries a zero start. These fits do not use IOLS; the estimates are roots of the estimating equations and do not depend on the solver that computes them.

A solution is reported only if it is a certified root of the original equation: in scaled coordinates, a score residual of at most $10^{-10}$ and a Newton correction of at most $10^{-9}$, the same bounds in the fixed metric, and a reciprocal condition estimate above $10^{-13}$. Standard errors additionally require the remaining numerical correction to be at most $10^{-6}$ standard errors.

Supplement S6 mentions arithmetic-mean equations at $\tau=0.5$, $1.5$ and $2.5$ in the replication files. They are not used in the paper's results, and this repository does not compute them. The reported GPML fits use the Gamma pilot and a zero fallback, without the $\tau$ continuation or shifted-log starts described in S6.

## Inference

Standard errors use the signed two-way cluster covariance

$$\hat V=\hat V_{\text{country}}+\hat V_{\text{family}}-\hat V_{\text{country}\times\text{family}},$$

where each term sums the outer products of the cluster totals of the observation-level influence contributions. For joint estimation, these contributions account for the estimated country effects. No finite-sample factor or eigenvalue adjustment is applied, so the matrix need not be positive semidefinite; every variance used for a reported interval or test is positive, and nonpositive variances and failed numerical checks would be reported with their status and a missing standard error. Intervals transform $b\pm1.96\,\widehat{\mathrm{SE}}(b)$ to $100(\exp(\cdot)-1)$, and p-values use the normal distribution, a many-cluster approximation with 48 countries and 94 ethnic families in the full sample.

Paired comparisons (PPML versus GPML, and half-power versus arithmetic mean) stack the influence contributions of both models on the same sample with the same cluster identifiers, so the covariance between the two estimates is included rather than assumed to be zero. Intervals and tests are pointwise and unadjusted for multiple comparisons, and paired tests compare models on a common sample, not estimates across deletion samples.

## Run

From the repository root, after installing `requirements.txt`:

```bash
python empirical/run.py --mode smoke --output outputs/application-smoke   # full sample only
python empirical/run.py --mode full --output outputs/full/empirical       # all six samples
```

`--data PATH` replaces the bundled data file, and `--resume` reuses completed samples of an interrupted run. Changing the data, the code, the dependencies or the selected samples requires a fresh output directory. The application needs neither the C++ IOLS module nor an internet connection. From the repository root, `python run.py --mode full --study empirical` refits the application and rebuilds its 13 tables, and `python run.py --mode report --study empirical --check-reference` rebuilds the tables from the saved results without refitting.

## Outputs

- `slopes.csv`: all 792 slope rows of a complete run, with standard errors in original units, intervals, inference status and the coefficients of the transformed-outcome models. Joint and cancelled PPML rows are both exported, so their equality can be checked.
- `country_intercepts.csv`: 2,002 country-effect point estimates of the joint and shifted-log fits. The original application did not export their standard errors; these fields are empty with status `not_exported`.
- `samples.csv`: sample sizes, zeros, cluster counts, and deletion and singleton counts.
- `covariance.csv`: the full signed covariance matrix of every specification.
- `paired_tests.csv` and `cross_covariance.csv`: the paired comparisons and their cross-model covariances.
- `run.json` and `sample_results/`: settings, numerical statuses and per-sample results used for resumption.

## Code

| File | Role |
|---|---|
| `run.py` | Refits all samples and writes the outputs |
| `config.json` | Solver settings, tolerances, outcome powers, working variances, shifts and samples |
| `fetch_data.py`, `row_order.json` | Rebuild the data file from the publisher's archive |
| `core/samples.py` | Complete-case, tail-deletion and singleton rules |
| `core/data.py` | Estimation sample and within-country design |
| `core/fits.py` | Safeguarded Newton roots for PPML and GPML |
| `core/power_mean.py` | Half-power fits and conversion to power-mean coefficients |
| `core/inference.py` | Influence contributions and the signed two-way cluster covariance |
| `core/cohort.py` | Adapts unequal country groups to the bundled solver's operators |
| `vendor/wz_fe_sim/` | The solver of the fixed-effect simulations (version 0.2.0) |
