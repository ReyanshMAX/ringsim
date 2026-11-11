"""`ringsim` command line (docs/ARCHITECTURE.md, CLI).

Every subcommand of ARCHITECTURE.md is declared here. Subcommands outside Phase 1 raise
NotImplementedError naming the BUILD.md phase that builds them. Subpackages are imported
inside the handlers only, so `ringsim --help` works with the base install.
"""
from __future__ import annotations

import argparse


def _not_built(name: str, phase: int) -> None:
    raise NotImplementedError(f"ringsim {name}: built in Phase {phase} (BUILD.md)")


CPU_MAX_CELLS = 96 ** 3          # D-062: sim without --gpu refuses Nx*Ny*Nz (excluding PML) above this
SCENE_PHASE = {"plate": 2, "hand": 3}


class CpuGridTooLarge(RuntimeError):
    """D-062 refusal: the grid is too large for a CPU run."""


def cmd_sim(args: argparse.Namespace) -> int:
    import math
    from pathlib import Path

    from ringsim.dataset.jobspec import load_job
    from ringsim.dataset.writer import ShardWriter
    from ringsim.devices.config import load_device
    from ringsim.geometry.scene import free_scene
    from ringsim.geometry.voxelize import SolidNearGridFace, build_medium, save_medium_png
    from ringsim.proc.physics import C_SIM
    from ringsim.solver.constants import DOMAIN_MARGIN_M
    from ringsim.solver.kwave_runner import medium_time_axis, run_ir

    job = load_job(args.job)
    if job.scene in SCENE_PHASE:
        _not_built(f"sim scene {job.scene}", SCENE_PHASE[job.scene])
    if job.scene != "free":
        raise ValueError(f"unknown scene {job.scene!r}")
    if args.frames is not None:
        raise NotImplementedError("--frames on a static scene: pending the lead (CLI Q-a)")
    dev = load_device(job.device)
    out_dir = Path(args.out_dir) if args.out_dir else Path("data") / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_id = "free"                                   # D-060

    shard = out_dir / f"{job.job_id}.h5"
    try:
        medium = build_medium(free_scene(dev), job.dx_m, C_SIM, DOMAIN_MARGIN_M)
    except SolidNearGridFace as e:                    # GEOMETRY.md rejection rule (D-053)
        with ShardWriter(str(shard), job, dev) as w:
            w.fail_sequence(seq_id, e.reason)
        print(f"{seq_id} failed: {e}", flush=True)
        return 1
    cells = math.prod(medium.rho.shape)
    if not args.gpu and cells > CPU_MAX_CELLS:
        raise CpuGridTooLarge(f"grid {medium.rho.shape} = {cells} cells > 96^3: too large for the CPU; run it on a "
                              f"T4 through `python tools/kaggle_run.py run {args.job}` (D-062)")
    save_medium_png(medium, out_dir / f"{job.job_id}_medium.png")
    _, n_t, k = medium_time_axis(medium, dev.range_window_m, dev.fs_hz)
    print(f"{job.job_id}: grid {medium.rho.shape}, dx {job.dx_m * 1e3:.2f} mm, n_t {n_t}, k {k}, "
          f"{'GPU' if args.gpu else 'CPU'}", flush=True)

    attrs = {"source": job.poses["source"], "n_frames": job.n_frames, "seed": job.seed}
    with ShardWriter(str(shard), job, dev) as w:
        w.begin_sequence(seq_id, attrs, None, n_t)

        def run_checked(m, what):
            """run_ir; a run-time exception or a non-finite IR fails the whole sequence (GEOMETRY.md)."""
            import numpy as np
            try:
                r = run_ir(m, dev.range_window_m, dev.fs_hz, args.gpu)
            except Exception as e:
                reason = f"run_error: {type(e).__name__}: {e}"
            else:
                if np.isfinite(r.ir).all():
                    print(f"{seq_id} {what}: wall_s {r.wall_s:.2f}", flush=True)
                    return r
                reason = "nonfinite_ir"
            w.fail_sequence(seq_id, reason)
            print(f"{seq_id} failed ({what}): {reason}", flush=True)
            return None

        res = run_checked(medium, "snapshot 0")
        if res is None:
            return 1
        w.write_snapshot(seq_id, 0, res, port_covered=medium.port_covered)
        # D-056: ref_ir is always its own call of the device-only scene, even where it equals the sim scene
        ref_medium = build_medium(free_scene(dev), job.dx_m, C_SIM, DOMAIN_MARGIN_M)
        ref = run_checked(ref_medium, "ref_ir")
        if ref is None:
            return 1
        w.write_ref_ir(seq_id, ref)
    print(f"wrote {shard}", flush=True)
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ringsim.dataset.reader import load_sequence
    from ringsim.validation.freefield import freefield_check

    fig, ax = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    for col, mode in enumerate(("raw", "ref")):
        s = load_sequence(args.shard, args.seq, None, None, None, mode=mode)
        for t in range(s.profiles.shape[0]):
            ax[0, col].plot(s.profiles[t], lw=0.8)
        ax[0, col].set_title(f"{args.seq}: {mode} profiles ({s.profiles.shape[0]} frames)")
        ax[0, col].set_xlabel("range bin")
        if s.diff.shape[0]:
            ax[1, col].imshow(s.diff.T, aspect="auto", origin="lower", cmap="RdBu_r")
        ax[1, col].set_title(f"{mode} differential [{s.diff.shape[0]} x {s.diff.shape[1]}]")
        ax[1, col].set_xlabel("frame"); ax[1, col].set_ylabel("range bin")
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    plt.close(fig)
    print(json.dumps(freefield_check(args.shard, args.seq), indent=1))   # D-060
    return 0


TX_FILE_STEM = {"iphone17pro": "phone"}      # <stem>_tx.wav, CAPTURE_APP.md "Files" (D-077)
SWEEP_F0_HZ, SWEEP_F1_HZ, SWEEP_DUR_S = 16000, 24000, 2.0   # sweep_16k_24k_2s.wav, VALIDATION.md Protocol A (D-077)


def cmd_export_tx(args: argparse.Namespace) -> int:
    """D-077: one call writes <stem>_tx.wav (the device yaml's chirp, one frame) and sweep_16k_24k_2s.wav."""
    from pathlib import Path

    from ringsim.devices.config import load_device
    from ringsim.proc.chirp import export_sweep_wav, export_tx_wav

    if args.device not in TX_FILE_STEM:
        raise ValueError(f"export-tx: no tx file name is specified for device {args.device!r} "
                         f"(CAPTURE_APP.md names {sorted(TX_FILE_STEM)})")
    dev = load_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tx = out / f"{TX_FILE_STEM[args.device]}_tx.wav"
    sweep = out / "sweep_16k_24k_2s.wav"
    export_tx_wav(dev.chirp, dev.fs_hz, str(tx))
    export_sweep_wav(SWEEP_F0_HZ, SWEEP_F1_HZ, SWEEP_DUR_S, dev.fs_hz, str(sweep))
    print(f"wrote {tx}", flush=True)
    print(f"wrote {sweep}", flush=True)
    return 0


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

    s = sub.add_parser("export-tx", help="write the device's tx WAV and the Protocol A sweep WAV")
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
