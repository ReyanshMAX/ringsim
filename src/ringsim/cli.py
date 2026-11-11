"""`ringsim` command line (docs/ARCHITECTURE.md, CLI).

Every subcommand of ARCHITECTURE.md is declared here. Subcommands outside Phase 1 raise
NotImplementedError naming the BUILD.md phase that builds them. Subpackages are imported
inside the handlers only, so `ringsim --help` works with the base install.
"""
from __future__ import annotations

import argparse


def _not_built(name: str, phase: int) -> None:
    raise NotImplementedError(f"ringsim {name}: built in Phase {phase} (BUILD.md)")


def cmd_sim(args: argparse.Namespace) -> int:
    raise NotImplementedError("ringsim sim: Phase 1, not built yet")


def cmd_profile(args: argparse.Namespace) -> int:
    raise NotImplementedError("ringsim profile: Phase 1, not built yet")


def cmd_export_tx(args: argparse.Namespace) -> int:
    _not_built("export-tx", 2)


def cmd_real(args: argparse.Namespace) -> int:
    _not_built("real", 2)


def cmd_validate(args: argparse.Namespace) -> int:
    _not_built(f"validate {args.what}", 2 if args.what == "plate" else 4)


def cmd_converge(args: argparse.Namespace) -> int:
    _not_built("converge", 3)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ringsim", description="ring-sim acoustic simulation pipeline")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="{sim,profile,export-tx,real,validate,converge}")

    s = sub.add_parser("sim", help="simulate a job into <out-dir>/<job_id>.h5")
    s.add_argument("--job", required=True, help="jobs/<job>.yaml")
    s.add_argument("--gpu", action="store_true", help="run k-Wave on the GPU")
    s.add_argument("--frames", metavar="a:b", help="frame range to simulate")
    s.add_argument("--out-dir", metavar="DIR", help="shard directory (default data/shards)")
    s.set_defaults(func=cmd_sim)

    s = sub.add_parser("profile", help="plot profiles + differential of one sequence")
    s.add_argument("--shard", required=True, help="<h5>")
    s.add_argument("--seq", required=True, help="sequence id")
    s.add_argument("--out", required=True, help="<png>")
    s.set_defaults(func=cmd_profile)

    s = sub.add_parser("export-tx", help="write the device's tx WAV (Phase 2)")
    s.add_argument("--device", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_export_tx)

    s = sub.add_parser("real", help="process a real capture (Phase 2)")
    s.add_argument("--capture", required=True, help="data/captures/<name>.wav")
    s.add_argument("--device", required=True)
    s.add_argument("--out", required=True, help="<png>")
    s.set_defaults(func=cmd_real)

    s = sub.add_parser("validate", help="sim vs real report (plate: Phase 2, hand: Phase 4)")
    s.add_argument("what", choices=["plate", "hand"])
    s.add_argument("--sim", required=True, help="<h5>")
    s.add_argument("--real", required=True, help="<dir>")
    s.add_argument("--out", required=True, help="reports/<name>/")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("converge", help="dx convergence study (Phase 3)")
    s.add_argument("--job", required=True, help="jobs/converge.yaml")
    s.add_argument("--dx", required=True, help="comma-separated dx in mm, e.g. 2.0,1.5,1.0")
    s.add_argument("--ref-dx", help="reference-check dx in mm (SOLVER.md)")
    s.set_defaults(func=cmd_converge)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
