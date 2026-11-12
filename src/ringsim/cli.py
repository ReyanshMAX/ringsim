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
SCENE_PHASE = {"hand": 3}


class CpuGridTooLarge(RuntimeError):
    """D-062 refusal: the grid is too large for a CPU run."""


def cmd_sim(args: argparse.Namespace) -> int:
    import math
    from pathlib import Path

    from ringsim.dataset.jobspec import load_job
    from ringsim.dataset.writer import ShardWriter
    from ringsim.devices.config import load_device
    from ringsim.geometry.scene import Placement, device_scene, free_scene, plate_scene
    from ringsim.geometry.voxelize import SolidNearGridFace, build_medium, save_medium_png
    from ringsim.proc.physics import C_SIM
    from ringsim.solver.constants import DOMAIN_MARGIN_M
    from ringsim.solver.kwave_runner import medium_time_axis, run_ir

    job = load_job(args.job)
    if job.scene in SCENE_PHASE:
        _not_built(f"sim scene {job.scene}", SCENE_PHASE[job.scene])
    if job.scene not in ("free", "plate"):
        raise ValueError(f"unknown scene {job.scene!r}")
    if args.frames is not None:
        raise NotImplementedError("--frames on a static scene: pending the lead (CLI Q-a)")
    dev = load_device(job.device)
    out_dir = Path(args.out_dir) if args.out_dir else Path("data") / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    if job.scene == "free":
        seq_id = "free"                               # D-060
        scene, ref_scene = free_scene(dev), free_scene(dev)
    else:
        placement = Placement.from_job(job.placement)
        seq_id = "plate" if placement.plate is not None else "background"   # D-071, D-078
        scene, ref_scene = plate_scene(dev, placement), device_scene(dev, placement)   # ref: body + rig (D-042)
    if job.poses["source"] != job.scene:
        raise ValueError(f"poses.source must be {job.scene!r} for scene {job.scene} (DATASET.md), "
                         f"got {job.poses['source']!r}")

    shard = out_dir / f"{job.job_id}.h5"
    try:
        medium = build_medium(scene, job.dx_m, C_SIM, DOMAIN_MARGIN_M)
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
        ref_medium = build_medium(ref_scene, job.dx_m, C_SIM, DOMAIN_MARGIN_M)
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


REAL_FS_HZ = 48000               # real iOS captures are 48 kHz, never resampled (PROCESSING.md Notes)


def cmd_real(args: argparse.Namespace) -> int:
    """One real capture through proc (PROCESSING.md Notes, D-070): own direct-path offset, frame 0 dropped, raw
    signed profiles. Plots them and prints JSON: the direct-path bin (CAPTURE_APP.md wants 0 with no target) and the
    frame-to-frame spread (metrics.noise_floor)."""
    import json
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from ringsim.devices.config import load_device
    from ringsim.proc.chirp import tx_frame
    from ringsim.proc.profile import align_direct_path, cfmcw_profiles, frame_stream, magnitude
    from ringsim.validation.metrics import noise_floor
    from ringsim.validation.real import load_capture

    dev = load_device(args.device)
    if dev.fs_hz != REAL_FS_HZ:
        raise ValueError(f"ringsim real: device {dev.name} runs at {dev.fs_hz} Hz; real captures are {REAL_FS_HZ} Hz")
    stream, meta = load_capture(args.capture)
    tx = tx_frame(dev.chirp, REAL_FS_HZ)
    band = (min(dev.chirp.f0_hz, dev.chirp.f1_hz), max(dev.chirp.f0_hz, dev.chirp.f1_hz))
    offset = align_direct_path(stream, tx, band, REAL_FS_HZ)
    frames = frame_stream(stream, offset, len(tx))[1:]          # frame 0: no echo tail from a previous period
    if len(frames) < 2:
        raise ValueError(f"{os.path.basename(args.capture)}: {len(frames)} frames after alignment and dropping "
                         "frame 0; need >= 2")
    prof = cfmcw_profiles(frames, tx, band, REAL_FS_HZ, dev.n_range_bins)
    mag = magnitude(prof)
    mean_mag = magnitude(prof.astype(np.float64).mean(axis=0)[None, :])[0]
    out = {"take": meta.get("take", os.path.splitext(os.path.basename(args.capture))[0]), "offset": int(offset),
           "n_frames": int(len(prof)), "direct_peak_bin": int(np.argmax(mean_mag)),
           "spread_raw": noise_floor(prof, skip_frames=0)}

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    for k in range(len(mag)):
        ax[0].plot(mag[k], color="0.7", lw=0.5)
    ax[0].plot(mean_mag, color="k", lw=1.5, label="mean profile")
    ax[0].set_xlabel("range bin"); ax[0].set_ylabel("|profile|"); ax[0].legend()
    ax[0].set_title(f"{out['take']}: raw profiles, {out['n_frames']} frames (frame 0 dropped), offset {offset}")
    im = ax[1].imshow(prof.T, aspect="auto", origin="lower", cmap="RdBu_r")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("range bin"); ax[1].set_title("signed profiles")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    plt.close(fig)
    print(json.dumps(out, indent=1))
    return 0


def _plate_report_files(rep: dict, sim: dict, real: dict, out) -> list[str]:
    """VALIDATION.md "Report format" for the plate: PNGs (mean profiles sim vs real; differential heatmaps side by
    side) and report.md with the metrics table. Every number comes from `rep` (validation.plate.plate_report)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from ringsim.proc.profile import differential

    ds = rep["distances_mm"]
    pngs = []
    for mode in ("ref", "raw"):
        fig, ax = plt.subplots(len(ds), 1, figsize=(9, 2.2 * len(ds)), squeeze=False)
        for i, d in enumerate(ds):
            ax[i, 0].plot(np.asarray(sim[mode][d], float).mean(0), label="sim")
            ax[i, 0].plot(np.asarray(real[mode][d], float).mean(0), label="real", alpha=0.8)
            ax[i, 0].set_title(f"{d} mm, {mode}, background-subtracted mean profile")
            ax[i, 0].set_xlabel("range bin")
            ax[i, 0].legend(fontsize=7)
        fig.tight_layout()
        name = f"mean_profiles_{mode}.png"
        fig.savefig(out / name, dpi=90)
        plt.close(fig)
        pngs.append(name)

        fig, ax = plt.subplots(len(ds), 2, figsize=(11, 2.2 * len(ds)), squeeze=False)
        for i, d in enumerate(ds):
            for j, (side, prof) in enumerate((("sim", sim[mode][d]), ("real", real[mode][d]))):
                df = differential(np.asarray(prof, np.float32))
                if len(df):
                    ax[i, j].imshow(df.T, aspect="auto", origin="lower", cmap="RdBu_r")
                    ax[i, j].set_title(f"{d} mm {side} {mode} differential [{df.shape[0]} x {df.shape[1]}]")
                else:
                    ax[i, j].set_title(f"{d} mm {side} {mode}: 1 frame, no differential")
                ax[i, j].set_xlabel("frame")
                ax[i, j].set_ylabel("range bin")
        fig.tight_layout()
        name = f"differential_{mode}.png"
        fig.savefig(out / name, dpi=90)
        plt.close(fig)
        pngs.append(name)

    def pf(b):
        return "PASS" if b else "FAIL"

    L = ["# Plate validation report (Protocol B)", "",
         f"temp_c {rep['temp_c']} | fs_hz {rep['fs_hz']} | distances_mm {ds} | takes per distance {rep['n_takes']}", "",
         "Metric 1 (peak bin error <= 1) is gated on referenced profiles (D-074); raw is reported with its own flag.", "",
         "| distance mm | mode | sim peak bin | real peak bin | error (bins) | signed-shape r | delta (bins) |",
         "|---|---|---|---|---|---|---|"]
    for mode in ("ref", "raw"):
        for d in ds:
            s = rep["signed_shape"][mode][d]
            L.append(f"| {d} | {mode} | {rep['peak_bin'][mode]['sim'][d]} | {rep['peak_bin'][mode]['real'][d]} | "
                     f"{rep['peak_bin_error'][mode][d]} | {s['r']:.4f} | {s['delta_bins']:.3f} |")
    L += ["", "| metric | mode | value | expected | result |", "|---|---|---|---|---|"]
    for mode in ("ref", "raw"):
        L.append(f"| peak bin error <= 1 at all distances | {mode} | | | {pf(rep['peak_bin_pass'][mode])} |")
        lin = rep["linearity"].get(mode)
        if lin:
            L.append(f"| range linearity slope (real), bins/m | {mode} | {lin['slope_bins_per_m']:.2f} | "
                     f"{lin['expected_bins_per_m']:.2f} (rel err {lin['rel_err']:.4f}) | {pf(lin['pass'])} |")
        at = rep["amp_trend"].get(mode)
        if at:
            L.append(f"| amplitude trend, log-log slope | {mode} | sim {at['sim']:.3f} / real {at['real']:.3f} | "
                     "report only | |")
    L += [f"| static noise floor n ({rep['noise_floor_basis']}) | ref | {rep['noise_floor']:.5f} | report only | |",
          f"| free-air drift, rel L2 end vs start | ref | {rep['free_air_drift']['rel_l2']:.5f} | report only | |",
          "", "Figures: " + ", ".join(f"[{n}]({n})" for n in pngs), ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    return pngs


def cmd_validate(args: argparse.Namespace) -> int:
    """`validate plate --sim <dir> --real <session dir> --out reports/<name>/` (D-078, VALIDATION.md Report format):
    report.md + PNGs + report.json from validation.plate. `validate hand` is Phase 4."""
    if args.what != "plate":
        _not_built(f"validate {args.what}", 4)
    import json
    from pathlib import Path

    from ringsim.validation.plate import plate_inputs, plate_report

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = plate_report(args.sim, args.real)
    sim, real, _, _, _ = plate_inputs(args.sim, args.real)
    _plate_report_files(rep, sim, real, out)
    js = json.dumps(rep, indent=1, default=float)
    (out / "report.json").write_text(js, encoding="utf-8")
    print(js)
    return 0


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

    s = sub.add_parser("real", help="process one real capture: profiles PNG + JSON")
    s.add_argument("--capture", required=True, help="data/captures/<name>.wav")
    s.add_argument("--device", required=True)
    s.add_argument("--out", required=True, help="<png>")
    s.set_defaults(func=cmd_real)

    s = sub.add_parser("validate", help="sim vs real report (plate: Phase 2, hand: Phase 4)")
    s.add_argument("what", choices=["plate", "hand"])
    s.add_argument("--sim", required=True, help="<dir> of plate shards + one background shard (D-078)")
    s.add_argument("--real", required=True, help="<dir> capture session (VALIDATION.md layout)")
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
