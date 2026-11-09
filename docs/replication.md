# Replication details

## Paper outputs

The main runner writes paper outputs into the `paper/` subdirectory of the selected output folder. LaTeX tables also have CSV exports for inspecting numerical values without LaTeX.

| Generated file | Results and numerical inputs |
|---|---|
| `solver_preliminary_main.tex` | Main IOLS C++ / Newton / IRLS comparison; failure counts and all-attempt capped median/mean CPU milliseconds |
| `solver_preliminary_appendix.tex` | NumPy IOLS, C++ IOLS, Newton and IRLS raw outcomes/timings for ordinary and stress designs |
| `solver_safeguarded_appendix.tex` | Damped Newton / L-BFGS-B raw outcomes and all-six-method common-budget failures |
| `solver_statistical_stress.tex` | Coefficient and normalized-effect summaries in the stress designs, with the paper's separate selection rules |
| `solver_stress_accuracy.pdf` | Larger stress designs; five displayed methods, two equations and two external accuracy thresholds |
| `solver_agreement.csv`, `solver_agreement_counts.json` | Supplement's within-equation coefficient/index comparisons; both accuracy gates, raw/common-budget views and the inspected-dataset exclusion |
| `fe_T2_display.tex`, `fe_T2_main_body.tex` | Main FE table at N=50/100/500, T=2; summary statistics from the completed draws |
| `fe_gpml_T2_compact.pdf` | GPML coefficient densities at N=20/50/100, T=2; all 5,000 draws enter each curve |
| `fe_wz_covariance_comparison_T2.tex` | Ordinary pair sandwich and WZ-adjusted covariance at the same WZ-corrected PPML coefficient |
| `fe_higher_order_covariance_pilot_T2.tex` | Separate first-100-draw covariance pilot at N=50/T=2; generic corrected coefficient held fixed |

The FE figure has 36 curves: four DGPs, three GPML methods and three sample sizes. It uses the original Silverman bandwidth rule and display-window calculation. Every draw contributes to the density; a plot window does not trim the sample. The exact final historical plotting script was unavailable, so the public renderer recreates the numerical content with its own typography/layout. Historical PDFs are supplied for comparison.

The solver accuracy figure displays C++ IOLS, statsmodels Newton, statsmodels IRLS, scikit-learn damped Newton and SciPy L-BFGS-B. Separate NumPy IOLS results remain in the simulation and appendix. Each displayed larger-stress group has 4,000 attempted fits in the completed study. Strict and secondary checks use 1e-8 and 1e-6.

## WZ design and method selection

The base experiment has four DGPs, N=20/50/100, T=2/5/10 and 5,000 replications per cell. Seed 20261002 generates a 100×100×10 master per replication, shared across the base cells. The N500/T2 extension has four DGPs and 5,000 replications each, with independent seed 20261004 and master shape 500×500×2. Matching replication IDs do not pair the extension with the base. The original Philox/raw-word Box–Muller recipe is bundled.

Historical source selection is explicit: 33,979 original successful base cases are retained, and the remaining base cases use the repaired Gamma-pilot admission implementation. The extension and covariance pilot use the repaired version. Both versions are included where required, with unchanged numerical branches. See the study README for the exact selection rule.

The main table labels map to saved method IDs GJ, G, GB, P and PW. PW uses the WZ-corrected PPML coefficient with the ordinary pair sandwich. PWa uses that same coefficient with WZ-adjusted covariance. The covariance pilot instead fixes the generic manuscript-corrected coefficient. Its first 100 base innovations per DGP and N50/T2 scope are separate from the main 5,000-draw experiment. Its higher-order variance is not clipped into validity.

Summary bootstraps use 999 resamples and seed 20261003. Dataset generation and reporting seeds have different roles. The supplied default saved arrays are limited to T=2, which covers the current paper's selected displays; full-run code retains the original additional T values.

## Solver design and interpretation

There are 16 ordinary and four stress designs, sizes (n,K)=(1,000,10) and (10,000,100), and 1,000 draws per cell. Each dataset is used for PPML and GPML with six methods. Seed 2026100301 and the original independent covariate/outcome/order streams are retained. A structural exclusion is recorded without redrawing the sample or dropping design columns.

The IOLS curvature step-grid version follows the paper's disclosed tuning on 20 inspected datasets; it is not untouched independent confirmation. Those dataset keys are retained for the relevant sensitivity calculations. Statsmodels, scikit-learn and SciPy versions are pinned, and the actual historical comparator routines are used.

Native solver status and the external equation check are different fields. The external check evaluates the original estimating equation and fitted-index correction with the same 1e-8/1e-6 thresholds. The completed study's later numerical verification retained 198 score-rejected index-diagnostic qualifications and two high-precision classification-verification qualifications. These did not alter the original acceptance flags. The small public runner does not import the private verification framework.

Historical raw costs, prospective termination and common-budget eligibility remain distinct. Default full reruns preserve which original calls were prospectively capped. A returned result retains its coefficient/native status even if late. Common-budget success additionally requires recorded CPU and elapsed times within 120 and 300 seconds. The main table reports all attempted CPU costs after clipping each to 120 seconds; appendix tables report original raw costs. Failures and overshoot remain represented in their proper denominators. Percentage formatting retains the original divide-then-multiply operation order.

Changing hardware, BLAS/compiler or floating-point environment can change runtimes and numerical boundary cases. Saved-result reporting rebuilds the completed paper results. Fresh reruns use the same designs, algorithms and settings, with their own recorded outcomes. Numerical examples and representative historical-value checks establish that the packaged paths run; they do not stand in for a newly executed full campaign.

## Environment and output checks

The checked local environment is macOS arm64, CPython 3.14.3, NumPy 2.4.2, SciPy 1.17.0, pandas 3.0.0, Matplotlib 3.10.8, scikit-learn 1.8.0 and statsmodels 0.15.0. The native build uses pybind11 3.0.2 and Eigen 5.0.1, with optimized release settings and precise floating-point flags. Numerical runners use POSIX process/timer facilities. Other platforms are not certified by the local check.

`--check-reference` compares numeric data cells in all seven data-bearing LaTeX files; the eighth is an FE-table wrapper. Both figures export their plotted numerical data alongside the PDF/PNG files. Regeneration never falls back from missing fresh results to the supplied original summaries.
