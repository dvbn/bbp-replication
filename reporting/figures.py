"""Regenerate the two paper figures from their saved numerical inputs."""
from __future__ import annotations

import math
from pathlib import Path

from common import percent, read_csv, require_unique, write_csv


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(figure, output, name):
    figure.savefig(output / (name + ".pdf"), bbox_inches="tight",
                   metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(output / (name + ".png"), dpi=160, bbox_inches="tight")


def solver_accuracy(inputs, output):
    from solver_tables import solver_folder
    folder = solver_folder(inputs)
    file = folder / "paper_accuracy_summary.csv"
    if file.is_file():
        rows = read_csv(file)
    else:
        rows = []
        for r in read_csv(folder / "outcomes.csv"):
            if r["variant"] == "full" and r["view"] == "common_cap" and r["scope"] == "pooled" and r["group"] == "stress" and int(r["n"]) == 10000:
                rows.append(dict(equation=r["estimator"], method=r["solver"], attempted=r["attempted"],
                                 strict_failures=int(r["attempted"]) - int(r["accurate"]),
                                 secondary_failures=int(r["attempted"]) - int(r["accepted_1e6"])))
    selected = require_unique(rows, lambda r: (r["equation"], r["method"]))
    methods = (("iols_cpp", "IOLS"), ("statsmodels_newton", "Newton"), ("statsmodels_irls", "IRLS"),
               ("sklearn_damped_newton", "Damped Newton"), ("scipy_lbfgs", "L-BFGS-B"))
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharex=True, sharey=True, layout="constrained")
    plotted = []
    for axis, equation in zip(axes, ("ppml", "gpml")):
        for i, (method, label) in enumerate(methods):
            row = selected[(equation, method)]
            attempted = int(row["attempted"])
            if attempted <= 0:
                raise ValueError("The accuracy figure has an empty attempted-fit denominator")
            strict = 100 * (int(row["strict_failures"]) / attempted)
            secondary = 100 * (int(row["secondary_failures"]) / attempted)
            axis.barh(i - .15, strict, height=.28, color="#31688e", label=r"$10^{-8}$ check" if i == 0 else None)
            axis.barh(i + .15, secondary, height=.28, color="#c8c8c8", edgecolor="#555555", linewidth=.3,
                      label=r"$10^{-6}$ check" if i == 0 else None)
            axis.text(min(strict + 1.2, 98), i - .15, percent(int(row["strict_failures"]), attempted),
                      va="center", fontsize=8, color="#233c4d")
            plotted.append(dict(equation=equation, method=method, attempted=attempted,
                strict_failures=int(row["strict_failures"]), secondary_failures=int(row["secondary_failures"]),
                strict_failure_percent=strict, secondary_failure_percent=secondary))
        axis.set_title(equation.upper(), fontsize=11)
        axis.set_yticks(range(len(methods)), [m[1] for m in methods])
        axis.set_xlim(0, 104)
        axis.set_xticks(range(0, 101, 20))
        axis.set_xlabel("Unsuccessful fits (%)")
        axis.grid(axis="x", color="#eeeeee")
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    save_figure(fig, output, "solver_stress_accuracy")
    plt.close(fig)
    write_csv(output / "solver_accuracy_plot_data.csv", plotted)


def quantile(sorted_values, p):
    index = p * (len(sorted_values) - 1)
    lo = math.floor(index)
    hi = min(lo + 1, len(sorted_values) - 1)
    fraction = index - lo
    return sorted_values[lo] * (1 - fraction) + sorted_values[hi] * fraction


def bandwidth(values):
    ordered = sorted(map(float, values))
    mean = math.fsum(ordered) / len(ordered)
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in ordered) / (len(ordered) - 1))
    iqr = quantile(ordered, .75) - quantile(ordered, .25)
    scale = min(sd, iqr / 1.349) if iqr > 0 else sd
    value = .9 * scale * len(ordered) ** (-.2)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("A density curve has an invalid Silverman bandwidth")
    return value


def density(values, grid, h):
    import numpy as np
    values = np.asarray(values, dtype=float)
    result = np.empty(len(grid))
    for start in range(0, len(grid), 32):
        z = (grid[start:start + 32, None] - values[None, :]) / h
        result[start:start + 32] = np.exp(-.5 * z * z).sum(axis=1)
    return result / (len(values) * h * math.sqrt(2 * math.pi))


def fe_densities(inputs, output):
    import numpy as np
    from wz_tables import load_arrays
    sizes = (20, 50, 100)
    methods = (("GJ", "Joint fixed effects"), ("G", "Cancellation"), ("GB", "Cancellation + correction"))
    arrays = {(dgp, n): load_arrays(inputs, n, dgp) for dgp in range(1, 5) for n in sizes}
    for a in arrays.values():
        for method, _ in methods:
            values = a[method + "_beta"]
            if len(values) < 2 or not np.isfinite(values).all():
                raise ValueError("Missing or nonfinite coefficients in the FE density inputs")
    plt = plotting()
    fig, axes = plt.subplots(3, 4, figsize=(11, 6.7), sharex="col", layout="constrained")
    styles = (("#2b6ca3", "--"), ("#d0781b", "-."), ("#1d1d1d", "-"))
    curves, metadata = [], []
    for dgp in range(1, 5):
        # Original display window: union of curve-wise central 99.8% intervals,
        # shared by GPML/PPML rows and sample sizes. Every draw enters each KDE.
        intervals = []
        for method in ("GJ", "G", "GB", "P", "PB", "PW"):
            for n in sizes:
                ordered = sorted(map(float, arrays[(dgp, n)][method + "_beta"]))
                intervals.append((quantile(ordered, .001), quantile(ordered, .999)))
        lo = min(1., min(x[0] for x in intervals))
        hi = max(1., max(x[1] for x in intervals))
        padding = .02 * (hi - lo)
        lo, hi = lo - padding, hi + padding
        grid = np.linspace(lo, hi, 401)
        for row, (method, label) in enumerate(methods):
            axis = axes[row, dgp - 1]
            for n, (color, linestyle) in zip(sizes, styles):
                values = arrays[(dgp, n)][method + "_beta"]
                h = bandwidth(values)
                y = density(values, grid, h)
                axis.plot(grid, y, color=color, linestyle=linestyle, linewidth=1.2, label=f"N = {n}")
                metadata.append(dict(dgp=dgp, method=method, N=n, T=2, draws=len(values), bandwidth=h,
                                     x_min=lo, x_max=hi, draws_outside_window=int(((values < lo) | (values > hi)).sum())))
                curves += [dict(dgp=dgp, method=method, N=n, x=float(x), density=float(v)) for x, v in zip(grid, y)]
            axis.axvline(1, color="#666666", linestyle=":", linewidth=.9)
            axis.set_ylim(bottom=0)
            axis.set_xlim(lo, hi)
            axis.tick_params(labelsize=8)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            if row == 0:
                axis.set_title(f"DGP {dgp}", fontsize=11)
            if dgp == 1:
                axis.set_ylabel(label + "\nDensity", fontsize=9)
            if row == 2:
                axis.set_xlabel("Estimated coefficient", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False, fontsize=9)
    save_figure(fig, output, "fe_gpml_T2_compact")
    plt.close(fig)
    write_csv(output / "fe_density_plot_data.csv", curves)
    write_csv(output / "fe_density_curves.csv", metadata)


def render(inputs, output, study="all"):
    if study in ("all", "solvers"):
        solver_accuracy(inputs, output)
    if study in ("all", "wz"):
        fe_densities(inputs, output)
