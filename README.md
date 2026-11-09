# Dealing with Logs and Zeros: Python replication

Replication files for **Dealing with Logs and Zeros in Regression Models**, by David Benatia, Christophe Bellégo and Louis-Daniel Pape.

This standalone repository contains the historical Python implementations for the paper's WZ fixed-effect simulations and six-method IOLS/solver comparison. The numerical kernels used for these studies are bundled. The general [powermean package](https://github.com/dvbn/powermean) is developed separately.

## Rebuild the paper tables and figures

Use Python 3.14. The checked environment uses CPython 3.14.3; dependency versions are in `requirements.txt`.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --mode report --check-reference
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Numerical simulation runners require Linux or macOS; report mode uses ordinary Python/NumPy/Matplotlib.

The command regenerates eight LaTeX table files, numerical CSV tables and two figures in PDF/PNG format under `outputs/report/paper/`. It checks the displayed table values against the supplied paper references. It performs no estimation and needs no compiler, LaTeX installation, account or external service. Only NumPy and Matplotlib are needed if installing dependencies just for this reporting step.

Saved inputs are restricted to the current displays. `reference/` contains expected paper tables and figures; `saved/` contains the numerical inputs used to regenerate them. The regenerated figures preserve the plotted numerical content with updated typography/layout.

## Run a small numerical example

Install CMake, a C++17 compiler, BLAS/LAPACK and Eigen headers for the native IOLS method. Build using the same Python environment:

```bash
python solvers/build_native.py --eigen-include /path/to/eigen3
python run.py --mode smoke
```

The Eigen directory must contain `Eigen/Core`. Eigen 3.4 or later is required; the checked local build uses Eigen 5.0.1. The script builds with one compilation job, optimized release settings and the historical floating-point flags. It does not download headers. Compiled binaries are built locally and are not distributed in this repository.

The smoke command runs a small WZ example, one covariance-pilot case, and both equations through all six solver methods. Results go to `outputs/smoke/`. A missing native module is reported explicitly. To run only WZ, which does not need this C++ module:

```bash
python run.py --mode smoke --study wz
```

## Rerun the simulations

```bash
python run.py --mode full
```

This is a substantial simulation campaign: 180,000 base WZ draw-cells, 20,000 independent N500 extension draw-cells, the separate 400-case covariance pilot, and 480,000 scheduled solver records across 40,000 datasets. Default execution is sequential with one BLAS/OpenMP thread per process. Completed case records are saved for resumption. Once both complete studies have finished, their own results are used to rebuild the displays under `outputs/full/paper/`.

To continue an interrupted run, repeat the same command with `--resume`. Keep the code, dependencies and scientific options unchanged. A fresh output directory is required for a different configuration.

Useful restricted runs:

```bash
python run.py --mode full --study wz --paper-only
python run.py --mode full --study solvers
python run.py --mode full --replications 2 --output outputs/small
```

`--paper-only` selects the WZ T=2 cells used in the current displays while retaining the original generator's master dimensions and seeds. An explicit replication limit creates a numerical subset and skips automatic paper reporting. Study-specific cell and method selectors are documented in the study READMEs. A subset is not the full Monte Carlo result.

For complete results previously generated in another directory:

```bash
python run.py --mode report --input-root outputs/full --output outputs/rebuilt
```

Original CPU timings are reproduced by the saved-result route. New simulation timings depend on the machine and environment. The full solver runner preserves the original distinction between prospectively capped calls and historical uncapped calls; the common 120 CPU / 300 elapsed-second reporting rule is applied separately.

## Contents and interpretation

- `wz/`: generators, historical numerical routines, configuration and covariance pilot. See [WZ instructions](wz/README.md).
- `solvers/`: six-method comparison, native build and scientific execution schedule. See [solver instructions](solvers/README.md).
- `reporting/`: small saved-result renderers.
- `saved/` and `reference/`: inputs and expected paper outputs.
- [Replication details](docs/replication.md): output map, seeds, method definitions and numerical qualifications.

Failures, structural exclusions and invalid inference are retained. Full reruns do not substitute a different estimator, relax the paper's acceptance thresholds, or silently replace C++ IOLS with NumPy.

The included authored code is released under [BSD-3-Clause](LICENSE). Applicable upstream notices are retained in the source and [third-party notices](THIRD_PARTY_NOTICES.md). Please cite the paper; citation metadata are in `CITATION.cff`.
