"""Fixed-effect tables from saved coefficients, standard errors and summaries."""
from __future__ import annotations

import math
from pathlib import Path
import statistics

from common import read_csv, require_unique, table, write_csv

METHODS = (("GJ", "GJ"), ("G", "GC"), ("GB", "GC+BC"), ("P", "PJ"), ("PW", "PJ+BC"))


def summary_rows(inputs):
    folder = Path(inputs) / "wz"
    rows = []
    for name in ("base", "n500"):
        rows += [r for r in read_csv(folder / name / "summary.csv") if int(r["T"]) == 2]
    return require_unique(rows, lambda r: (int(r["dgp"]), int(r["N"]), r["method_id"]))


def main_table(rows, output):
    body = [r"\begingroup", r"\small", r"\setlength{\tabcolsep}{2pt}",
            r"\begin{tabular}{@{}rl*{4}{rrr}@{}}", r"\toprule",
            r" & & \multicolumn{3}{c}{DGP 1} & \multicolumn{3}{c}{DGP 2} & \multicolumn{3}{c}{DGP 3} & \multicolumn{3}{c}{DGP 4}\\",
            r"$N$ & Method & Bias & SE & Cov. & Bias & SE & Cov. & Bias & SE & Cov. & Bias & SE & Cov.\\", r"\midrule"]
    exported = []
    for n in (50, 100, 500):
        for i, (method, label) in enumerate(METHODS):
            values = []
            for dgp in range(1, 5):
                row = rows[(dgp, n, method)]
                bias = float(row["bias_over_sd"])
                se = float(row["mean_se_over_sd"])
                coverage = 100 * float(row["coverage_cr"])
                values += [f"{bias:.2f}", f"{se:.2f}", f"{coverage:.1f}"]
                exported.append(dict(N=n, T=2, dgp=dgp, method=method,
                    bias_over_sd=bias, mean_se_over_sd=se, coverage_percent=coverage,
                    replications=int(row["R"]), intervals=int(row["n_cov"])))
            body.append((str(n) if i == 0 else "") + " & " + label + " & " + " & ".join(values) + r"\\")
        body.append(r"\addlinespace[2pt]")
    body += [r"\bottomrule", r"\end{tabular}", r"\endgroup", ""]
    (output / "fe_T2_main_body.tex").write_text("\n".join(body))
    wrapper = "\n".join([r"\begin{table}[p]", r"\centering",
        r"\caption{Centering, standard-error accuracy and coverage with fixed $T=2$}",
        r"\label{tab:mc-fe-main-T2}", r"\input{fe_T2_main_body.tex}",
        r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize\raggedright",
        r"Bias is bias divided by empirical SD; SE is mean standard error divided by empirical SD; Cov. is 95\% normal-interval coverage. "
        "GJ is joint GPML; GC is cancellation; GC+BC adds the generic coefficient correction. PJ is PPML and PJ+BC uses the WZ-corrected coefficient with the ordinary pair sandwich. "
        "The independent N=500 extension is not paired with the smaller-size draws.",
        r"\end{minipage}\end{table}", ""])
    (output / "fe_T2_display.tex").write_text(wrapper)
    write_csv(output / "fe_main.csv", exported)


def load_arrays(inputs, n, dgp):
    import numpy as np
    folder = Path(inputs) / "wz" / ("n500" if n == 500 else "base")
    with np.load(folder / f"dgp{dgp}_N{n}_T2.npz", allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def metrics(beta, variance):
    beta, variance = list(map(float, beta)), list(map(float, variance))
    if len(beta) != len(variance) or len(beta) < 2:
        raise ValueError("A covariance comparison needs matching draws and at least two coefficients")
    if not all(math.isfinite(x) for x in beta) or not all(math.isfinite(x) and x > 0 for x in variance):
        raise ValueError("Missing or nonpositive covariance input; refusing to silently condition the comparison")
    se = [math.sqrt(x) for x in variance]
    covered = [int(abs(b - 1) <= 1.96 * s) for b, s in zip(beta, se)]
    return dict(R=len(beta), available=len(se), mean_se_over_sd=statistics.mean(se) / statistics.stdev(beta),
                coverage_percent=100 * statistics.mean(covered), covered=covered)


def covariance_table(inputs, output):
    import numpy as np
    body, exported = [], []
    for n in (50, 100, 500):
        for dgp in range(1, 5):
            arrays = load_arrays(inputs, n, dgp)
            if not np.array_equal(arrays["PW_beta"], arrays["PWa_beta"]):
                raise ValueError("PW/PWa covariance columns must use the same coefficient")
            vals = {method: metrics(arrays[method + "_beta"], arrays[method + "_se"] ** 2)
                    for method in ("PW", "PWa")}
            a, b = vals["PW"], vals["PWa"]
            if a["available"] != b["available"]:
                raise ValueError("Covariance comparisons use different available samples")
            body.append(f"{dgp} & {n} & {a['mean_se_over_sd']:.3f} & {a['coverage_percent']:.1f} & "
                        f"{b['mean_se_over_sd']:.3f} & {b['coverage_percent']:.1f} & {a['available']:,}" + r"\\")
            exported.append(dict(dgp=dgp, N=n, T=2, replications=a["R"], available=a["available"],
                pair_se_over_sd=a["mean_se_over_sd"], pair_coverage_percent=a["coverage_percent"],
                wz_se_over_sd=b["mean_se_over_sd"], wz_coverage_percent=b["coverage_percent"]))
        body.append(r"\addlinespace[3pt]")
    text = table(r"PPML covariance comparison, $T=2$", "tab:fe-wz-covariance-comparison", "rrrrrrr",
        r"DGP & $N$ & Pair SE/SD & Pair Cov. (\%) & WZ SE/SD & WZ Cov. (\%) & Avail.", body,
        "Both columns use the same WZ-corrected PPML coefficient. The ordinary pair sandwich has no finite-pair multiplier; "
        "WZ adjusted includes its finite-sample adjustment. SE/SD is mean standard error divided by empirical SD (denominator R-1). "
        r"Cov. is nominal 95\% normal-interval coverage. The N=500 extension is independent of the smaller draws.")
    (output / "fe_wz_covariance_comparison_T2.tex").write_text(text)
    write_csv(output / "fe_wz_covariance.csv", exported)


def pilot_table(inputs, output):
    rows = read_csv(Path(inputs) / "wz/pilot/draw_results.csv")
    selected = require_unique(rows, lambda r: (int(r["dgp"]), float(r["tau"]), int(r["rep"])))
    body, exported = [], []
    for tau, equation, panel in ((1.0, "PPML", "A"), (2.0, "GPML", "B")):
        body.append(r"\multicolumn{6}{@{}l}{\textit{Panel " + panel + ": Cancelled " + equation + r", generic coefficient correction}}\\")
        for dgp in range(1, 5):
            current = sorted((r for k, r in selected.items() if k[:2] == (dgp, tau)), key=lambda r: int(r["rep"]))
            if len(current) < 2:
                raise ValueError("Missing covariance-pilot cell")
            beta = [float(r["beta_bc"]) for r in current]
            base = metrics(beta, [float(r["variance_raw"]) for r in current])
            alternatives = [("Pair sandwich", "variance_raw"), ("Higher order", "variance_adjusted_raw")]
            if tau == 1:
                alternatives.append(("WZ adjusted", "variance_wz_adjusted"))
            for label, key in alternatives:
                result = metrics(beta, [float(r[key]) for r in current])
                change = [a-b for a, b in zip(result["covered"], base["covered"])]
                delta = 100 * statistics.mean(change)
                mcse = 100 * statistics.stdev(change) / math.sqrt(len(current))
                delta_text = "ref." if key == "variance_raw" else f"{delta:.0f} ({mcse:.2f})"
                body.append(f"{dgp} & {label} & {result['mean_se_over_sd']:.3f} & {result['coverage_percent']:.0f} & "
                            f"{delta_text} & {result['available']}/{result['R']}" + r"\\")
                exported.append(dict(dgp=dgp, tau=tau, covariance=label, replications=result["R"],
                    available=result["available"], mean_se_over_sd=result["mean_se_over_sd"],
                    coverage_percent=result["coverage_percent"], paired_change_pp=delta, paired_change_mcse_pp=mcse))
            body.append(r"\addlinespace[2pt]")
    text = table(r"Exploratory covariance comparison: first saved draws, $N=50$, $T=2$", "tab:fe-higher-order-pilot", "llrrrr",
        r"DGP & Covariance & SE/SD & Cov. (\%) & $\Delta$ (MCSE) & Avail.", body,
        r"Every covariance within an equation uses the same generic corrected coefficient, not the separate WZ coefficient correction. "
        r"Higher order uses $\widehat V_2=\widehat V_0-\widehat\Delta_n$ without clipping, PSD projection or finite-pair multiplier. "
        "SE/SD uses the pilot coefficients' empirical SD. Delta is paired coverage change in percentage points; MCSE is the sample SD of "
        "indicator differences divided by the square root of the draw count. Zero observed MCSE is not zero population uncertainty. "
        "This exploratory pilot is not a coverage or covariance-MSE guarantee.")
    (output / "fe_higher_order_covariance_pilot_T2.tex").write_text(text)
    write_csv(output / "fe_covariance_pilot.csv", exported)


def render(inputs, output):
    main_table(summary_rows(inputs), output)
    covariance_table(inputs, output)
    pilot_table(inputs, output)
