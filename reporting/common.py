"""Small helpers shared by the paper display scripts."""
from __future__ import annotations

import csv
import math
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return None
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"Nonfinite {key} in a saved summary")
    return x


def require_unique(rows, key):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError(f"Duplicate saved-summary key: {k}")
        result[k] = row
    return result


def table(caption, label, columns, header, body, note):
    return "\n".join([
        r"\begin{table}[p]", r"\centering", r"\small",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}" + columns + r"@{}}", r"\toprule",
        header + r"\\", r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}", r"\par\smallskip",
        r"\begin{minipage}{\linewidth}\footnotesize\raggedright",
        r"\textit{Notes:} " + note, r"\end{minipage}", r"\end{table}", "",
    ])


def percent(failures, attempted):
    if not attempted:
        return "--"
    # The paper's original operation order matters at binary half-way values.
    return format(100 * (failures / attempted), ".2f")
