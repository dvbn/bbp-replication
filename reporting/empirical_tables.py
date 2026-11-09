"""Tables for political centralization and nighttime light density.

Read unrounded scientific exports. No estimation or data selection takes place
here. Half-power coefficients and standard errors are already on gamma/omega
scale in slopes.csv.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
from statistics import NormalDist

from common import read_csv, require_unique, table, write_csv

TABLE_NAMES = ("application_sensitivity.tex", "application_all_coefficients.tex",
               "application_full_grid.tex", "application_full_intervals.tex", "application_positive_only.tex")
VARIABLES = (
    ("centr_tribe", "Jurisdictional hierarchy"), ("lnpdmean00s", "Log population density"),
    ("capdistance1", "Distance to capital"), ("seadist1", "Distance to coast"),
    ("borderdist1", "Distance to border"), ("lnwaterkm", "Log water area"),
    ("lnkm2split", "Log land area"), ("mean_elev", "Mean elevation"),
    ("mean_suit", "Agricultural suitability"), ("malariasuit", "Malaria suitability"),
    ("petroleum", "Petroleum indicator"), ("diamondd", "Diamond indicator"),
)
SAMPLES = (
    ("full", "Full", "full sample"),
    ("bright_top1row", "Drop 1", "largest outcome value excluded"),
    ("bright_top1pct", r"Drop 1\%", r"brightest 1\% excluded"),
    ("bright_top2pct", r"Drop 2\%", r"brightest 2\% excluded"),
    ("bright_top5pct", r"Drop 5\%", r"brightest 5\% excluded"),
    ("positive_full", r"\(Y>0\)", "positive outcomes only"),
)
OLS_IDS = ("ols_c1e-05", "ols_c0.01", "ols_c1")
POWER_IDS = ("joint_omega1_tau1", "joint_omega1_tau2", "cancelled_omega1_tau2",
             "joint_omega0.5_tau1", "joint_omega0.5_tau2", "cancelled_omega0.5_tau2")
MODELS = (
    (OLS_IDS[0], r"OLS, \(c=10^{-5}\)", "Abs."),
    (OLS_IDS[1], r"OLS, \(c=0.01\)", "Abs."),
    (OLS_IDS[2], r"OLS, \(c=1\)", "Abs."),
    (POWER_IDS[0], r"PPML, \(Y\)", "J=C"),
    (POWER_IDS[1], r"GPML, \(Y\)", "J"),
    (POWER_IDS[2], r"GPML, \(Y\)", "C"),
    (POWER_IDS[3], r"Poisson, \(\sqrt{Y}\)", "J=C"),
    (POWER_IDS[4], r"Gamma, \(\sqrt{Y}\)", "J"),
    (POWER_IDS[5], r"Gamma, \(\sqrt{Y}\)", "C"),
)
Z95 = NormalDist().inv_cdf(.975)


def finite(value):
    if value in ("", None):
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def estimate(row):
    beta = finite(row["coefficient"]) if row["fit_status"] == "ok" else None
    se = finite(row["standard_error_cr0"])
    if (beta is None or se is None or se <= 0
            or row["inference_status"] not in ("ok", "not_applicable_closed_form")):
        return dict(beta=beta, se=None, p=None, stars="", lo=None, hi=None)
    p = math.erfc(abs(beta / se) / math.sqrt(2))
    lo, hi = finite(row.get("ci95_lo")), finite(row.get("ci95_hi"))
    if lo is None or hi is None:
        lo, hi = beta - Z95 * se, beta + Z95 * se
    return dict(beta=beta, se=se, p=p, stars="***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "", lo=lo, hi=hi)


def starred(value, stars, digits):
    if value is None:
        return "--"
    return f"{value:.{digits}f}" + ("^{" + stars + "}" if stars else "")


def coefficient_cell(row):
    e = estimate(row)
    beta = starred(e["beta"], e["stars"], 3)
    se = "--" if e["se"] is None else f"{e['se']:.3f}"
    return r"\BBPcoefcell{" + beta + "}{" + se + "}"


def association(row):
    e = estimate(row)
    transform = lambda x: None if x is None else 100 * math.expm1(x)
    return e | dict(percent=transform(e["beta"]), lower_percent=transform(e["lo"]), upper_percent=transform(e["hi"]))


def p_text(p):
    return "--" if p is None else "<0.001" if p < .001 else f"{p:.3f}"


def association_cell(row, macro="BBPappcell"):
    a = association(row)
    return "\\" + macro + "{" + starred(a["percent"], a["stars"], 2) + "}{" + p_text(a["p"]) + "}"


def interval(row):
    a = association(row)
    return "--" if a["lower_percent"] is None else rf"\([{a['lower_percent']:.2f}, {a['upper_percent']:.2f}]\)"


def observation_row(sample_ids, samples, empty_fe=False):
    return "Observations & " + (" & " if empty_fe else "") + " & ".join(str(samples[s]["n_obs"]) for s in sample_ids) + r"\\"


def scientific_note(associations=True):
    return (("Entries are percentage associations per hierarchy category. " if associations else "") +
        r"Two-sided normal tests use unrounded coefficients and standard errors: "
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^*p<0.10$. J denotes joint country effects, C cancellation, and Abs. OLS absorption. "
        r"All models include the eleven controls and use signed country-and-ethnic-family two-way CR0 covariance, without a finite-sample "
        r"or eigenvalue adjustment. " +
        (r"Square-root models use $\widehat\gamma/\omega$ and correspondingly rescaled standard errors before $100(\exp(\cdot)-1)$ transformation. " if associations else "") +
        "Missing inference is shown by a dash, not a fabricated standard error or significance star.")


def render_main(lookup, samples, output):
    chosen = ("full", "bright_top2pct", "bright_top5pct", "positive_full")
    body = [observation_row(chosen, samples)]
    for fid, label, fe in MODELS[:6]:
        short_label = label if fid in OLS_IDS else "PPML" if fid == POWER_IDS[0] else f"GPML ({fe})"
        body.append(short_label + " & " + " & ".join(association_cell(lookup[s, fid, "centr_tribe"], "BBPappmaincell") for s in chosen) + r"\\")
    text = r"\providecommand{\BBPappmaincell}[2]{\(#1\;(#2)\)}" + "\n" + table(
        "Political centralization and nighttime light density", "table:michalopoulos", "lcccc",
        r"Specification & Full & Drop 2\% & Drop 5\% & \(Y>0\)", body,
        scientific_note() + " Full and trimmed samples retain zeros; the positive-only sample removes zeros and reapplies the country-singleton rule.")
    (output / TABLE_NAMES[0]).write_text(text)


def render_coefficients(lookup, samples, output):
    pieces = [r"\providecommand{\BBPcoefcell}[2]{\shortstack{\(#1\)\\(\(#2\))}}"]
    sample_ids = [s[0] for s in SAMPLES]
    header = "Regressor & " + " & ".join(s[1] for s in SAMPLES)
    for i, (fid, shift) in enumerate(zip(OLS_IDS, (r"10^{-5}", "0.01", "1"))):
        body = [label + " & " + " & ".join(coefficient_cell(lookup[s, fid, variable]) for s in sample_ids) + r"\\"
                for variable, label in VARIABLES]
        body += [observation_row(sample_ids, samples), "Country effects & " + " & ".join(["Absorbed"] * 6) + r"\\"]
        pieces.append(table(r"Shifted-log slope coefficients: \(c=" + shift + r"\)",
            "table:application-coefficients-ols" + ("" if i == 0 else f"-{i+1}"), "lcccccc", header, body,
            r"OLS coefficients for $\log(Y+c)$, with signed two-way CR0 standard errors in parentheses. "
            "Coefficients are in original regressor units, not percentage changes. " + scientific_note(False)))
    header = (r" & \multicolumn{3}{c}{Arithmetic mean ($\omega=1$)} & \multicolumn{3}{c}{Half-power mean ($\omega=1/2$)}\\" + "\n" +
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n" +
              r"Score index & $\tau=1$ & $\tau=2$ & $\tau=2$ & $\tau=1$ & $\tau=2$ & $\tau=2$\\" + "\n" +
              r"Regressor / country FE & J=C & J & C & J=C & J & C")
    for sid, _, caption in SAMPLES:
        body = [label + " & " + " & ".join(coefficient_cell(lookup[sid, fid, variable]) for fid in POWER_IDS) + r"\\"
                for variable, label in VARIABLES]
        body.append("Observations & " + " & ".join([str(samples[sid]["n_obs"])] * 6) + r"\\")
        pieces.append(table("Arithmetic- and half-power-mean slope coefficients: " + caption,
            "table:application-coefficients-" + sid.replace("_", "-"), "lcccccc", header, body,
            r"Entries are $\widehat\beta_\omega=\widehat\gamma/\omega$ with correspondingly rescaled signed two-way CR0 standard errors. "
            r"At $\tau=1$, joint and cancelled slopes and standard errors coincide. No coefficient-bias correction is applied. "
            "The positive-only sample changes the population. " + scientific_note(False)))
    (output / TABLE_NAMES[1]).write_text("\n".join(pieces))


def render_supplement(lookup, samples, output):
    chosen = [s[0] for s in SAMPLES[:5]]
    header = r"Specification & FE & Full & Drop 1 & Drop 1\% & Drop 2\% & Drop 5\%"
    body = [observation_row(chosen, samples, empty_fe=True)]
    for fid, label, fe in MODELS:
        body.append(label + " & " + fe + " & " + " & ".join(association_cell(lookup[s, fid, "centr_tribe"]) for s in chosen) + r"\\")
    text = r"\providecommand{\BBPappcell}[2]{\shortstack{\(#1\)\\(\(#2\))}}" + "\n" + table(
        r"Five-sample sensitivity results: percentage associations and $p$-values", "table:application-full-grid", "llccccc", header, body,
        scientific_note() + " Deletions remove the original-ranked 1, 7, 14 and 35 largest outcome values; the last also removes a country singleton. All five samples retain all 164 zeros.")
    (output / TABLE_NAMES[2]).write_text(text)
    body = [label + " & " + fe + " & " + " & ".join(interval(lookup[s, fid, "centr_tribe"]) for s in chosen) + r"\\"
            for fid, label, fe in MODELS]
    (output / TABLE_NAMES[3]).write_text(table(r"Five-sample sensitivity results: 95\% confidence intervals", "table:application-full-intervals",
        "llccccc", header, body,
        r"The percentage-association intervals transform the normal coefficient interval endpoints through $100(\exp(\cdot)-1)$. "
        "Half-power coefficients and standard errors are both rescaled by two before transformation. " + scientific_note()))
    body = []
    for fid, label, fe in MODELS:
        a = association(lookup["positive_full", fid, "centr_tribe"])
        body.append(label + " & " + fe + r" & \(" + starred(a["percent"], a["stars"], 2) + r"\) & " +
                    interval(lookup["positive_full", fid, "centr_tribe"]) + r" & \(" + p_text(a["p"]) + r"\)\\")
    sample = samples["positive_full"]
    (output / TABLE_NAMES[4]).write_text(table("Positive-outcome-only sample: percentage associations and inference",
        "table:application-positive-only", "llccc", r"Specification & FE & Estimate & 95\% interval & $p$-value", body,
        f"The sample contains {sample['n_obs']} observations in {sample['n_countries']} countries and {sample['n_families']} ethnic families "
        "after excluding zeros and a resulting country singleton. No upper-tail cut is applied. " + scientific_note()))


def render(inputs: Path, output: Path):
    folder, output = Path(inputs) / "empirical", Path(output)
    rows = read_csv(folder / "slopes.csv")
    lookup = require_unique(rows, lambda r: (r["sample_id"], r["fit_id"], r["variable"]))
    samples = require_unique(read_csv(folder / "samples.csv"), lambda r: r["sample_id"])
    for sid, _, _ in SAMPLES:
        for fid in OLS_IDS + POWER_IDS:
            for variable, _ in VARIABLES:
                row = lookup[sid, fid, variable]
                if int(row["n_obs"]) != int(samples[sid]["n_obs"]):
                    raise ValueError("Application coefficient and sample counts disagree")
        for omega in (1, .5):
            for variable, _ in VARIABLES:
                a = lookup[sid, f"joint_omega{omega:g}_tau1", variable]
                b = lookup[sid, f"cancelled_omega{omega:g}_tau1", variable]
                for key in ("coefficient", "standard_error_cr0"):
                    x, y = finite(a[key]), finite(b[key])
                    if (x is None) != (y is None) or (x is not None and not math.isclose(x, y, rel_tol=2e-10, abs_tol=2e-11)):
                        raise ValueError("Joint and cancelled tau=1 exports violate their exact identity")
    output.mkdir(parents=True, exist_ok=True)
    render_main(lookup, samples, output)
    render_coefficients(lookup, samples, output)
    render_supplement(lookup, samples, output)
    associations = []
    for sid, _, _ in SAMPLES:
        for fid, _, fe in MODELS:
            row = lookup[sid, fid, "centr_tribe"]
            a = association(row)
            associations.append(dict(sample_id=sid, fit_id=fid, country_effects=fe, n_obs=row["n_obs"],
                coefficient=a["beta"], standard_error_cr0=a["se"], p_value_normal=a["p"], stars=a["stars"],
                percent_association=a["percent"], lower_percent=a["lower_percent"], upper_percent=a["upper_percent"]))
    write_csv(output / "application_associations.csv", associations)
    countries = read_csv(folder / "country_intercepts.csv")
    return dict(tex_files=list(TABLE_NAMES), printed_tables=13, raw_slope_rows=len(rows),
                distinct_printed_coefficient_cells=648, country_intercept_rows=len(countries),
                unavailable_inference_rows=sum(estimate(r)["se"] is None for r in rows), model_fits=0)


def table_payload(name, text):
    """Read printed data cells and stars independently from the renderer."""
    starts = tuple(label for _, label in VARIABLES) + ("OLS,", "PPML", "GPML", "Poisson,", "Gamma,", "Observations")
    result = []
    for line in text.splitlines():
        line = re.split(r"(?<!\\)%", line, maxsplit=1)[0].strip()
        if not line.startswith(starts) or "&" not in line or not line.endswith(r"\\"):
            continue
        cells = line[:-2].split("&")
        start = 1 if name in TABLE_NAMES[:2] else 2
        payload = []
        for cell in cells[start:]:
            cell = re.sub(r"\s+", "", cell).replace(r"\(", "").replace(r"\)", "")
            payload.append(cell)
        result.append(payload)
    return result


def verify(output: Path, reference: Path):
    checks = []
    for name in TABLE_NAMES:
        actual = table_payload(name, (Path(output) / name).read_text())
        expected = table_payload(name, (Path(reference) / name).read_text())
        if not expected or actual != expected:
            differences = [(i+1, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
            raise ValueError(f"Application reference differs: {name}; rows {len(actual)}/{len(expected)}; {differences[:2]}")
        checks.append(dict(table=name, data_rows=len(actual), printed_cells=sum(map(len, actual)), matches=True))
    return checks
