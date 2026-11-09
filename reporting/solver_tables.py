"""Rebuild solver tables from scalar summaries, without fitting models."""
from __future__ import annotations

from decimal import Decimal, localcontext
import json
from pathlib import Path

from common import number, percent, read_csv, require_unique, table, write_csv

METHODS = ("iols_numpy", "iols_cpp", "statsmodels_newton", "statsmodels_irls",
           "sklearn_damped_newton", "scipy_lbfgs")
LABELS = dict(zip(METHODS, ("IOLS, NumPy", "IOLS", "Newton", "IRLS", "Damped Newton", "L-BFGS-B")))
MAIN = ("iols_cpp", "statsmodels_newton", "statsmodels_irls")
PANELS = tuple((group, n, equation) for group in ("ordinary", "stress")
               for n in (1000, 10000) for equation in ("ppml", "gpml"))
DESIGNS = {
    "S1_dispersion25_nb2_strong": "Negative binomial",
    "S2_dispersion25_lognormal25": "Lognormal",
    "S3_contamination_p001_x500": "Contamination",
    "S4_equicorrelated999_lognormal": "Near-collinearity",
}


def solver_folder(inputs):
    folder = Path(inputs) / "solvers"
    return folder / "summary" if (folder / "summary").is_dir() else folder


def pooled_outcomes(folder):
    rows = [r for r in read_csv(Path(folder) / "outcomes.csv")
            if r["variant"] == "full" and r["scope"] == "pooled"]
    return require_unique(rows, lambda r: (r["view"], r["group"], int(r["n"]), r["estimator"], r["solver"]))


def milliseconds(x):
    return f"{x:.1f}" if abs(x) < 10 else f"{x:,.0f}"


def main_table(folder, output):
    rows = read_csv(Path(folder) / "paper_main_summary.csv")
    selected = require_unique(rows, lambda r: (r["group"], int(r["n"]), r["equation"], r["method"]))
    body, exported = [], []
    for group, n, equation in PANELS:
        values = []
        for method in MAIN:
            row = selected[(group, n, equation, method)]
            attempted = int(row["attempted"])
            failure = percent(int(row["strict_failures"]), attempted)
            median = number(row, "capped_cpu_median_ms")
            mean = number(row, "capped_cpu_mean_ms")
            timing = "--" if median is None or mean is None else milliseconds(median) + "/" + milliseconds(mean)
            values += [failure, timing]
            exported.append(dict(group=group, n=n, equation=equation, method=method,
                attempted=attempted, failures=int(row["strict_failures"]),
                failure_percent=failure, median_cpu_ms=median, mean_cpu_ms=mean,
                displayed_time_ms=timing))
        body.append(f"{group.title()}, {n:,} & {equation.upper()} & " + " & ".join(values) + r"\\")
    header = (r" & & \multicolumn{2}{c}{IOLS} & \multicolumn{2}{c}{Newton} & \multicolumn{2}{c}{IRLS}\\" + "\n" +
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}" + "\n" +
        r"Design, $n$ & Equation & Fail \% & \shortstack{Time (ms)\\med./mean} & Fail \% & \shortstack{Time (ms)\\med./mean} & Fail \% & \shortstack{Time (ms)\\med./mean}")
    text = table("Solver comparison under a common computational budget", "tab:solver-preliminary-main",
        "llrcrcrc", header, body,
        r"The three method pairs are IOLS C++, statsmodels Newton and statsmodels IRLS. "
        r"Failure means no output passing the common $10^{-8}$ check within 120 CPU and 300 wall seconds. "
        "CPU median/mean uses all attempted fits, including failures, with each recorded CPU time clipped at 120 seconds. "
        "Setup and certified structural exclusions are omitted. Historical timing results depend on the original hardware.")
    (output / "solver_preliminary_main.tex").write_text(text)
    write_csv(output / "solver_main.csv", exported)


def appendices(outcomes, output):
    files = {}
    exported = []
    for group in ("ordinary", "stress"):
        body = []
        for g, n, equation in PANELS:
            if g != group:
                continue
            first = outcomes[("raw", g, n, equation, "iols_cpp")]
            body.append(r"\multicolumn{9}{@{}l}{\textit{" +
                f"{g.title()}, $n={n:,}$, {equation.upper()}: {int(first['attempted']):,} attempts; " +
                f"{int(first['structural']):,} structural exclusions" + r"}}\\")
            for method in METHODS[:4]:
                row = outcomes[("raw", g, n, equation, method)]
                values = raw_cells(row)
                body.append(LABELS[method] + " & " + " & ".join(values) + r"\\")
                exported.append(dict(group=g, n=n, equation=equation, method=method,
                    accurate=values[0], audit_miss=values[1], other=values[2], limit=values[3],
                    median_cpu_seconds=values[4], mean_cpu_seconds=values[5],
                    p95_cpu_seconds=values[6], maximum_cpu_seconds=values[7]))
            body.append(r"\addlinespace[2pt]")
        label = "tab:solver-preliminary-appendix" if group == "ordinary" else "tab:solver-preliminary-stress"
        files[group] = table("Solver outcomes and CPU time: " + group + " designs", label,
            "lrrrrrrrr", "Solver & Accurate & Audit miss & Other & Limit & Median & Mean & 95th pct. & Max.", body,
            outcome_note())
    (output / "solver_preliminary_appendix.tex").write_text(files["ordinary"] + "\n" + files["stress"])
    body = []
    for group, n, equation in PANELS:
        first = outcomes[("raw", group, n, equation, "iols_cpp")]
        body.append(r"\multicolumn{9}{@{}l}{\textit{" +
            f"{group.title()}, $n={n:,}$, {equation.upper()}: {int(first['attempted']):,} attempts; " +
            f"{int(first['structural']):,} structural exclusions" + r"}}\\")
        for method in METHODS[4:]:
            body.append(LABELS[method] + " & " + " & ".join(raw_cells(outcomes[("raw", group, n, equation, method)])) + r"\\")
        body.append(r"\addlinespace[2pt]")
    text = table("Safeguarded solver outcomes and CPU time", "tab:solver-safeguarded-appendix",
        "lrrrrrrrr", "Solver & Accurate & Audit miss & Other & Limit & Median & Mean & 95th pct. & Max.", body, outcome_note())
    body = []
    for group, n, equation in PANELS:
        cells = []
        for method in METHODS:
            row = outcomes[("common_cap", group, n, equation, method)]
            cells.append(percent(int(row["attempted"]) - int(row["accurate"]), int(row["attempted"])))
        body.append(f"{group.title()}, {n:,} & {equation.upper()} & " + " & ".join(cells) + r"\\")
    text += "\n" + table("Unsuccessful fits under the common computing budget (percent)", "tab:solver-all-six-common-cap",
        "llrrrrrr", "Design, $n$ & Equation & IOLS NumPy & IOLS & Newton & IRLS & Damped Newton & L-BFGS-B", body,
        r"Percentages use attempted fits, excluding certified structural failures. Accuracy requires the common $10^{-8}$ "
        "check and observed CPU and wall times no greater than 120 and 300 seconds. Historical over-budget calls and completed "
        "overruns are unsuccessful under this rule, not relabeled as observed timeouts.")
    (output / "solver_safeguarded_appendix.tex").write_text(text)
    write_csv(output / "solver_raw_outcomes.csv", exported)


def raw_cells(row):
    keys = ("accurate", "outcome_native_success_audit_miss", "outcome_native_failure")
    values = [f"{int(row[k]):,}" for k in keys]
    values.append(f"{int(row['outcome_budget_limit']) + int(row['outcome_time_budget_exhausted']):,}")
    for stat in ("median", "mean", "p95", "maximum"):
        x = number(row, "raw_cpu_" + stat)
        values.append("--" if x is None else f"{x:.4f}")
    return values


def outcome_note():
    return (r"Accurate fits pass the common $10^{-8}$ check; audit misses are native successes failing that check. "
        "Other is the remaining native or numerical failure category. Limit combines internal iteration-limit failures and observed timeouts. "
        "CPU seconds include every attempted call, including failures and timeouts, and exclude generation, setup, external auditing and "
        "the separately recorded interrupted attempt. These are raw recorded costs. Quantiles use the inverse empirical CDF. "
        "IOLS is C++; NumPy is separate. Newton/IRLS use statsmodels; damped Newton uses scikit-learn and L-BFGS-B uses SciPy.")


def stress_value(row, field, stat, digits, scale=1):
    raw = row.get(field + "_" + stat, "")
    if raw in ("", None):
        overflow = json.loads(row.get(field + "_arithmetic_overflow_fields", "[]"))
        return r"\(\mathrm{OVF}\)" if stat in overflow else "--"
    with localcontext() as context:
        context.prec = 80
        x = Decimal(str(raw)) * Decimal(scale)
        if not x.is_finite():
            raise ValueError("Nonfinite stress statistic")
        if x and (abs(x) >= 10000 or abs(x) < Decimal(10) ** -digits):
            mantissa, exponent = format(x, ".2E").split("E")
            return r"\(" + mantissa + r"\!\times\!10^{" + str(int(exponent)) + r"}\)"
        return format(x, f".{digits}f")


def stress_table(folder, output):
    rows = [r for r in read_csv(Path(folder) / "statistics.csv") if r["variant"] == "full" and r["view"] == "common_cap"
            and r["scope"] == "cell" and r["group"] == "stress" and r["solver"] == "iols_cpp"
            and r["sample"] in ("own_strict", "cpp_strict_statsmodels_fail")]
    selected = require_unique(rows, lambda r: (int(r["n"]), r["dgp"], r["estimator"], r["sample"]))
    body, exported = [], []
    for sample, title in (("own_strict", "Panel A. Each equation's accurate IOLS fits"),
                          ("cpp_strict_statsmodels_fail", "Panel B. GPML: IOLS accurate, Newton and IRLS unsuccessful")):
        body.append(r"\multicolumn{7}{@{}l}{\textbf{" + title + r"}}\\")
        for n in (1000, 10000):
            body.append(r"\multicolumn{7}{@{}l}{\textit{" + f"$n={n:,}$, $K={10 if n==1000 else 100}$" + r"}}\\")
            for dgp, label in DESIGNS.items():
                for equation in (("ppml", "gpml") if sample == "own_strict" else ("gpml",)):
                    row = selected[(n, dgp, equation, sample)]
                    counts = [int(row[k]) for k in ("selected", "beta_n", "theta_n")]
                    draws = f"{counts[0]:,}" if len(set(counts)) == 1 else "/".join(f"{x:,}" for x in counts)
                    vals = [draws, stress_value(row, "beta", "bias", 4), stress_value(row, "beta", "sd", 4),
                            stress_value(row, "theta", "bias", 2, 100), stress_value(row, "theta", "sd", 2, 100),
                            stress_value(row, "theta", "median", 2, 100)]
                    name = label + (", " + equation.upper() if sample == "own_strict" else "")
                    body.append(name + " & " + " & ".join(vals) + r"\\")
                    exported.append(dict(n=n, dgp=dgp, equation=equation, sample=sample,
                        draws=draws, beta_bias=vals[1], beta_sd=vals[2], theta_bias_pp=vals[3],
                        theta_sd_pp=vals[4], theta_median_percent=vals[5]))
    text = table("Statistical behavior of accurate IOLS fits in the stress designs", "tab:solver-statistical-stress", "lrrrrrr",
        r"Design / equation & Draws & Bias & SD & Bias (pp) & SD (pp) & Median (\%)", body,
        r"True treatment coefficient is $\log(1.08)$ and true normalized ATE is 8\%. All estimates use C++ IOLS and pass "
        r"the common $10^{-8}$ check within 120 CPU and 300 elapsed seconds. Panel A uses each equation's own accurate fits; "
        "Panel B additionally requires both statsmodels comparators to be unsuccessful. Each coefficient is transformed by "
        r"$\exp(\widehat\beta_D)-1$ before target summaries. Bias is mean minus truth; SD is empirical standard deviation. "
        "No accurate finite estimate is trimmed. Medians use the inverse empirical CDF; a dash denotes an empty selection.")
    (output / "solver_statistical_stress.tex").write_text(text)
    write_csv(output / "solver_stress_statistics.csv", exported)


def render(inputs, output):
    folder = solver_folder(inputs)
    main_table(folder, output)
    outcomes = pooled_outcomes(folder)
    appendices(outcomes, output)
    stress_table(folder, output)
    agreement_results(folder, output)


def agreement_results(folder, output):
    """Export the within-equation agreement results reported in the supplement."""
    rows = read_csv(Path(folder) / "agreement_summary.csv")
    require_unique(rows, lambda r: (r["selection"], r["view"], r["gate"], r["cell_id"],
                                   r["estimator"], r["solver_a"], r["solver_b"]))
    with (Path(folder) / "agreement_counts.json").open() as stream:
        counts = json.load(stream)
    for selection, name in (("full", "pair_rows"),
                            ("exclude_inspected20", "pairs_after_inspected_exclusion")):
        actual = sum(int(r["available_dataset_pairs"]) for r in rows
                     if r["selection"] == selection and r["view"] == "raw" and r["gate"] == "strict")
        if actual != int(counts[name]):
            raise ValueError(f"Pairwise agreement count differs for {selection}: {actual}")
    write_csv(output / "solver_agreement.csv", rows)
    (output / "solver_agreement_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
