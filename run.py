#!/usr/bin/env python3
"""Run the BBP simulations or rebuild the paper displays from saved results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("report", "smoke", "full"), default="report")
    parser.add_argument("--study", choices=("all", "wz", "solvers"), default="all")
    parser.add_argument("--output", type=Path, help="Output directory; default: outputs/MODE")
    parser.add_argument("--input-root", type=Path, help="Saved-result root for report mode")
    parser.add_argument("--replications", type=positive_int, help="Limit each cell to this many replications; save numerical results without automatic reporting")
    parser.add_argument("--paper-only", action="store_true", help="WZ: run only T=2 cells")
    parser.add_argument("--skip-pilot", action="store_true", help="WZ: omit the separate covariance pilot")
    parser.add_argument("--resume", action="store_true", help="Continue an existing numerical run")
    parser.add_argument("--tables-only", action="store_true", help="Skip figure rendering")
    parser.add_argument("--check-reference", action="store_true", help="Compare saved-result tables with the paper")
    args = parser.parse_args()
    if args.mode != "report" and args.input_root is not None:
        parser.error("--input-root is only used with --mode report")
    if args.mode != "report" and args.check_reference:
        parser.error("--check-reference applies to the original saved-result report")
    if args.mode == "report" and (args.replications is not None or args.resume or args.paper_only or args.skip_pilot):
        parser.error("--replications, --resume, --paper-only and --skip-pilot are numerical-run options")
    output = (args.output or ROOT / "outputs" / args.mode).expanduser().resolve()
    protected = [ROOT / p for p in ("saved", "reference", "wz", "solvers", "reporting", "docs")]
    if output == ROOT or any(output.is_relative_to(p) for p in protected):
        parser.error("choose an output directory outside the supplied source and saved inputs")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # The historical comparisons use a single BLAS/OpenMP thread per worker.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(output / ".matplotlib")

    def run(command: list[str]) -> None:
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)

    try:
        if args.mode != "report":
            studies = ("wz", "solvers") if args.study == "all" else (args.study,)
            for study in studies:
                command = [str(ROOT / study / "run.py"), "--mode", args.mode, "--output", str(output / study)]
                if args.replications is not None:
                    command.extend(("--replications", str(args.replications)))
                if args.resume:
                    command.append("--resume")
                if study == "wz":
                    if args.paper_only:
                        command.append("--paper-only")
                    if args.skip_pilot:
                        command.append("--skip-pilot")
                run(command)
            if args.mode == "smoke":
                print(f"Small numerical run finished. Results: {output}")
                return 0
            if args.replications is not None or args.skip_pilot:
                print(f"Selected numerical run finished. Results: {output}")
                print("Automatic paper reporting is reserved for complete runs. Use --mode report --input-root with complete reporting inputs.")
                return 0
        inputs = (args.input_root or ROOT / "saved").expanduser().resolve() if args.mode == "report" else output
        command = [str(ROOT / "reporting" / "render.py"), "--input-root", str(inputs), "--output", str(output / "paper"), "--study", args.study]
        if args.tables_only:
            command.append("--tables-only")
        if args.check_reference:
            command.append("--check-reference")
        run(command)
    except subprocess.CalledProcessError as exc:
        print(f"A replication step failed (exit {exc.returncode}). See its message above; completed results remain in {output}.", file=sys.stderr)
        return exc.returncode if 0 < exc.returncode < 126 else 1
    print(f"Paper outputs: {output / 'paper'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
