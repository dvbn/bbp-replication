# Political centralization and nighttime lights

This application revisits the nighttime-light specification in Table III, column 4 of Michalopoulos and Papaioannou (2013), *Econometrica* 81(1), 113–152, [doi:10.3982/ECTA9613](https://doi.org/10.3982/ECTA9613). The observation is an ethnic homeland within a country. The original 12 regressors, country effects and country/ethnic-family clustering are retained. The source is the authors' [original replication package](https://www.econometricsociety.org/publications/econometrica/2013/01/01/pre%E2%80%90colonial-ethnic-institutions-and-contemporary-african/supp/9613_data_and_programs_0.zip).

The bundled CSV contains the necessary columns and original row order. See [data attribution and reconstruction](../DATA_AVAILABILITY.md). A fresh copy can also be reconstructed from the original article supplement:

```bash
python empirical/fetch_data.py --output outputs/data/recovered-data.csv
```

## Run

From the repository root, after installing the numerical requirements:

```bash
python empirical/run.py --mode smoke --output outputs/application-smoke
python empirical/run.py --mode full --output outputs/full/empirical
```

The default data file is `empirical/data/greek_econometrica_data.csv`. `--data PATH` supplies an explicit replacement. Smoke mode refits all requested specifications on the full sample. Full mode refits all six sensitivity samples. Use `--resume` with the same command to reuse completed samples; changed data, numerical source, dependencies or sample scope require a fresh output directory.

## Samples and models

The complete-case rule uses the outcome, all 12 regressors, and both cluster keys. Country singletons are removed. Each restriction begins from the same original baseline; country-singleton removal, centering, column scales, estimation and inference are recomputed.

| Sample | Observations |
|---|---:|
| Full | 682 |
| Remove brightest observation | 681 |
| Remove brightest 1% | 675 |
| Remove brightest 2% | 668 |
| Remove brightest 5% | 646 |
| Positive outcomes only | 517 |

Brightness uses the raw outcome, descending, with original source-row position breaking ties. Percentage cuts use ceiling counts from all 682 baseline observations, including zeros. The 5% restriction removes 35 selected observations and one resulting country singleton. The positive-only restriction removes 164 zeros and one resulting singleton.

Each sample estimates PPML and GPML on `Y` and `sqrt(Y)`, with joint country effects and the separate cancelled-country equations. Power-mean coefficients are `gamma/omega`; corresponding influences and standard errors are divided by `omega`. The three shifted-log OLS specifications use `log(Y+c)` with `c=0.00001, 0.01, 1` and absorbed country effects. Joint and cancelled PPML share the exact profile identity; both export rows are retained for checking it. Cancelled GPML remains a separate specification.

The historical numerical core uses safeguarded Newton with the actual Jacobian, LU solves, a fixed merit and the original certificate and fallback-start rules. The Gamma pilot followed by C2 and C1 is retained. Unreported later score orders are omitted; the current paper uses score indices 1 and 2. This application runner is independent of the powermean package and does not describe these Newton fits as iOLS.

## Inference and outputs

The main covariance is the signed CR0 sum of country and ethnic-family cluster meats minus their intersection meat. It is not eigenvalue-clipped. Nonpositive variances and failed numerical checks retain their explicit status and missing standard errors. Normal 95% coefficient intervals use `norm.ppf(0.975)`; percentage associations transform their endpoints using `100*(exp(coefficient)-1)`.

The output directory contains:

- `slopes.csv`: all 792 coefficient rows for a complete run, with original-unit standard errors, intervals, inference statuses and the separate transformed-outcome coefficients.
- `country_intercepts.csv`: 2,002 joint/shifted-log country-intercept point estimates. The historical application did not export their standard errors; those fields remain empty with status `not_exported`.
- `samples.csv`: sample sizes, zeros, cluster counts and deletion/singleton counts.
- `covariance.csv`: full signed covariance matrices for every specification.
- `paired_tests.csv` and `cross_covariance.csv`: paired PPML–GPML and half-power–arithmetic coefficient comparisons. Cross-outcome covariance uses stacked row influences on identical samples and the same cluster keys. Independent-variance sums are not substituted.
- `run.json` and `sample_results/`: scientific settings, numerical statuses and completed-sample results used for resumption.

These are actual refits. The repository's report mode separately rebuilds the current paper tables from the supplied saved results. Outcome restrictions are sensitivity exercises; they do not select a preferred sample. Reported intervals and paired tests retain the paper's pointwise conventions.
