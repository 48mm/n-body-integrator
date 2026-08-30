"""
N-body studies.

Each study writes its numbers as a CSV and its figure as a PNG to output.

Studies that need solar-system initial conditions use JPL Horizons by default,
which are cached on disk, and fall back to an offline demo system if the network 
is unavailable.

``--source demo`` forces offline,
``--source jpl`` forces live service.
"""

import argparse
import os

import matplotlib

import analysis

# Each study maps to a function in analysis.py.
STUDIES = {
    "twobody": analysis.two_body,
    "conservation": analysis.conservation,
    "longterm": analysis.long_term,
    "validation": analysis.validation,
    "chaos": analysis.chaos,
    "benchmark": analysis.benchmark,
}
NEEDS_SOURCE = {"conservation", "longterm", "validation", "chaos", "benchmark"}


def run_study(name, args):
    print("=" * 70)
    print(f"STUDY: {name}")
    print("=" * 70)
    kwargs = {"outdir": args.outdir, "show": not args.no_show}
    if name in NEEDS_SOURCE:
        kwargs["source"] = args.source
    if args.dt is not None and name != "benchmark":
        kwargs["dt"] = args.dt
    STUDIES[name](**kwargs)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Numerical N-body integration study (Euler / Euler-Cromer / RK4).")
    parser.add_argument("study", choices=list(STUDIES) + ["all"],
                        help="Which study to run ('all' runs every study).")
    parser.add_argument("--source", choices=["jpl", "demo", "auto"], default="auto",
                        help="Initial-condition source (default: auto).")
    parser.add_argument("--outdir", default="results",
                        help="Directory for CSV and PNG output (default: results).")
    parser.add_argument("--dt", type=float, default=None,
                        help="Time step in seconds (default: per-study report value).")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures without opening a window for batch runs.")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    os.makedirs(args.outdir, exist_ok=True)

    studies = list(STUDIES) if args.study == "all" else [args.study]
    for name in studies:
        try:
            run_study(name, args)
        except Exception as err:
            print(f"  study {name!r} failed: {err}\n")
            if args.study != "all":
                raise SystemExit(1)
    print(f"Done. Output written to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
