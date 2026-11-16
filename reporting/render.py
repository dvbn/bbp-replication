#!/usr/bin/env python3
"""Rebuild the paper tables and figures from saved numerical results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
BUNDLE = HERE.parent
TABLES = {
    "wz": ("fe_T2_main_body.tex", "fe_T2_display.tex", "fe_wz_covariance_comparison_T2.tex",
           "fe_higher_order_covariance_pilot_T2.tex"),
    "solvers": ("solver_preliminary_main.tex", "solver_preliminary_appendix.tex",
                "solver_safeguarded_appendix.tex", "solver_statistical_stress.tex"),
    "empirical": ("application_sensitivity.tex", "application_all_coefficients.tex",
                  "application_full_grid.tex", "application_full_intervals.tex", "application_positive_only.tex"),
}


def numeric_rows(name, text):
    """Read displayed data cells, excluding captions, headings and notes."""
    groups = []
    inside, pending = False, ""
    for raw in text.splitlines():
        line = re.split(r"(?<!\\)%", raw, maxsplit=1)[0].strip()
        if r"\begin{tabular}" in line:
            inside = True
            continue
        if r"\end{tabular}" in line:
            inside, pending = False, ""
            continue
        if not inside or not line:
            continue
        if line.startswith((r"\toprule", r"\midrule", r"\bottomrule", r"\cmidrule", r"\addlinespace", r"\multicolumn")):
            continue
        if "&" not in line and not pending:
            continue
        pending += " " + line
        if not line.endswith(r"\\"):
            continue
        row, pending = pending.strip()[:-2].strip(), ""
        cells = [c.strip() for c in row.split("&")]
        first = cells[0]
        start = None
        if name == "solver_preliminary_main.tex":
            if first.startswith(("Ordinary", "Stress")) or (not first and len(cells) > 1 and cells[1] in ("PPML", "GPML")):
                start = 2
        elif name in ("solver_preliminary_appendix.tex", "solver_safeguarded_appendix.tex"):
            if first.startswith(("IOLS", "Newton", "IRLS", "Damped Newton", "L-BFGS-B")):
                start = 1
            elif first.startswith(("Ordinary", "Stress")):
                start = 2
        elif name == "solver_statistical_stress.tex":
            if first.startswith(("Negative binomial", "Lognormal", "Contamination", "Near-collinearity")):
                start = 1
        elif name == "fe_T2_main_body.tex":
            if len(cells) > 1 and cells[1] in ("GJ", "GC", "GC+BC", "PJ", "PJ+BC"):
                start = 2
        elif name in ("fe_wz_covariance_comparison_T2.tex", "fe_higher_order_covariance_pilot_T2.tex"):
            if first.isdecimal():
                start = 0
        if start is not None:
            result = []
            for cell in cells[start:]:
                cell = cell.replace(",", "").replace("$-$", "-").replace("$+$", "+")
                nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", cell)
                result.append(nums if nums else ["OVF" if "OVF" in cell else "--" if "--" in cell else cell])
            groups.append(result)
    return groups


def verify_tables(output, reference, names=None):
    checks = []
    files = sorted(Path(output).glob("*.tex")) if names is None else [Path(output) / name for name in names]
    for file in files:
        target = Path(reference) / file.name
        if not target.is_file() or file.name == "fe_T2_display.tex":
            continue
        actual = numeric_rows(file.name, file.read_text())
        expected = numeric_rows(file.name, target.read_text())
        if not expected:
            raise ValueError(f"No independently readable reference data rows in {file.name}")
        checks.append(dict(table=file.name, rows=len(actual), matches=actual == expected))
        if actual != expected:
            mismatches = [dict(row=i+1, actual=a, reference=b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
            raise ValueError(f"Reference values differ in {file.name}: rows {len(actual)}/{len(expected)}; {mismatches[:3]}")
    if not checks:
        raise ValueError("No output tables were checked")
    return checks


def run(input_root, output, study="all", tables_only=False, check_reference=None):
    inputs, output = Path(input_root).resolve(), Path(output).resolve()
    if (output == inputs or inputs.is_relative_to(output)
            or any(output.is_relative_to(inputs / name) for name in TABLES)):
        raise ValueError("Reporting output must be separate from its input tree")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    if study in ("all", "solvers"):
        import solver_tables
        solver_tables.render(inputs, output)
    if study in ("all", "wz"):
        import wz_tables
        wz_tables.render(inputs, output)
    empirical = None
    if study in ("all", "empirical"):
        import empirical_tables
        empirical = empirical_tables.render(inputs, output)
    if not tables_only:
        import figures
        figures.render(inputs, output, study)
    names = [name for group, files in TABLES.items() if study in ("all", group) for name in files]
    figure_count = sum(study in ("all", group) for group in ("wz", "solvers"))
    result = dict(study=study, tables=len(names),
                  figures=0 if tables_only else figure_count, model_fits=0)
    if empirical is not None:
        result["empirical"] = empirical
    if check_reference is not None:
        simulation_names = [name for group, files in TABLES.items()
                            if group != "empirical" and study in ("all", group) for name in files]
        checks = verify_tables(output, check_reference, simulation_names) if simulation_names else []
        if empirical is not None:
            checks += [row | dict(rows=row["data_rows"]) for row in empirical_tables.verify(output, check_reference)]
        result["reference_checks"] = checks
    (output / "display_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", "--inputs-root", type=Path, default=BUNDLE / "saved")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", choices=("all", "wz", "solvers", "empirical"), default="all")
    parser.add_argument("--tables-only", action="store_true", help="Skip density and accuracy figures")
    parser.add_argument("--check-reference", nargs="?", const=BUNDLE / "reference/tables", type=Path,
                        help="Compare generated numeric table cells with the paper reference tables")
    args = parser.parse_args(argv)
    result = run(args.input_root, args.output, args.study, args.tables_only, args.check_reference)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
