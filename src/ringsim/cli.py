"""`ringsim` entry point (ARCHITECTURE.md CLI). Phase 1: sim, profile."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

CPU_MAX_DIM = 96            # INFRA.md: CPU only for dims <= 96^3 (excluding PML)


def _data_root() -> Path:
    return Path(os.environ.get("RINGSIM_DATA", "data"))


def save_medium_png(medium, path: Path) -> None:
    """3-slice PNG of rho (xy/xz/yz through the mic voxel), GEOMETRY.md Notes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i, j, k = medium.mic_idx[0]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, sl, title in ((ax[0], medium.rho[:, :, k].T, f"xy @ z={k}"),
                         (ax[1], medium.rho[:, j, :].T, f"xz @ y={j}"),
                         (ax[2], medium.rho[i, :, :].T, f"yz @ x={i}")):
        im = a.imshow(sl, origin="lower", cmap="gray_r", vmin=0, vmax=1000)
        a.set_title(title)
    for (si, sj, sk) in medium.src_idx:
        ax[0].plot(si, sj, "r^"); ax[1].plot(si, sk, "r^")
    for (mi, mj, mk) in medium.mic_idx:
        ax[0].plot(mi, mj, "bo"); ax[1].plot(mi, mk, "bo"); ax[2].plot(mj, mk, "bo")
    fig.colorbar(im, ax=ax, shrink=0.7, label="rho (kg/m^3)")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def cmd_sim(args) -> int:
    from ringsim.dataset.jobspec import load_job
    from ringsim.dataset.writer import ShardWriter
    from ringsim.devices.config import load_device
    from ringsim.geometry.scene import free_scene
    from ringsim.geometry.voxelize import build_medium
    from ringsim.proc.physics import C_SIM
    from ringsim.solver.constants import DOMAIN_MARGIN_M
    from ringsim.solver.grid import time_axis
    from ringsim.solver.kwave_runner import run_ir

    job = load_job(args.job)
    dev = load_device(job.device)
    if job.scene != "free":
        print(f"scene {job.scene!r} not implemented yet (Phase 2+)", file=sys.stderr)
        return 2
    n_frames = int(job.poses["n_frames"])
    a, b = 0, n_frames
    if args.frames:
        a, b = (int(x) if x else d for x, d in zip(args.frames.split(":"), (0, n_frames)))
    out_dir = Path(args.out_dir) if args.out_dir else _data_root() / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / f"{job.job_id}.h5"

    scene = free_scene(dev)
    medium = build_medium(scene, job.dx_m, C_SIM, DOMAIN_MARGIN_M)
    if not args.gpu and max(medium.rho.shape) > CPU_MAX_DIM:
        print(f"grid {medium.rho.shape} > {CPU_MAX_DIM}^3: too large for CPU, "
              f"use `ringsim kaggle push --job {args.job}`", file=sys.stderr)
        return 2
    save_medium_png(medium, out_dir / f"{job.job_id}_medium.png")
    _, n_t, _ = time_axis(job.dx_m, dev.range_window_m, dev.fs_hz)
    seq_id = "free"
    attrs = {"source": "free", "fps": 1.0 / dev.chirp.frame_period_s,
             "ring_rot_deg": 0.0, "ring_shift_mm": 0.0, "seed": job.seed}
    print(f"{job.job_id}: grid {medium.rho.shape}, dx {job.dx_m} m, n_t {n_t}, frames {a}:{b}")
    with ShardWriter(str(shard), job, dev) as w:
        w.begin_sequence(seq_id, attrs, n_frames, n_t)
        for t in range(a, b):
            res = run_ir(medium, dev.range_window_m, dev.fs_hz, args.gpu)  # free: same medium every frame
            w.write_frame(seq_id, t, res)
            print(f"frame {t}: wall_s {res.wall_s:.2f}")
    print(f"wrote {shard}")
    return 0


def cmd_profile(args) -> int:
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml

    from ringsim.dataset.reader import list_sequences, load_sequence
    from ringsim.proc import magnitude

    seq = args.seq or list_sequences(args.shard)[0]
    s = load_sequence(args.shard, seq, snr_db=None, temp_c=args.temp_c, rng=None)
    with h5py.File(args.shard, "r") as f:
        job = yaml.safe_load(f.attrs["job_yaml"])
    out = Path(args.out or Path(args.shard).with_name(f"{Path(args.shard).stem}_{seq}_profile.png"))
    T, nb = s.profiles.shape
    ncol = 3 if T > 1 else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5 * ncol, 4))
    ax[0].plot(s.profiles[0], label="signed"); ax[0].plot(magnitude(s.profiles)[0], label="|hilbert|")
    ax[0].set_xlabel("range bin"); ax[0].set_title(f"{seq} frame 0"); ax[0].legend()
    ax[1].imshow(s.profiles.T, aspect="auto", origin="lower"); ax[1].set_title("profiles [bin x frame]")
    if T > 1:
        ax[2].imshow(s.diff.T, aspect="auto", origin="lower", cmap="RdBu"); ax[2].set_title("differential")
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)
    report = {"seq": seq, "frames": T, "n_bins": nb, "offset": s.meta["offset"],
              "peak_bin": int(np.argmax(magnitude(s.profiles)[0]))}
    if job["scene"] == "free":
        from ringsim.validation.freefield import freefield_check
        report["freefield"] = freefield_check(args.shard, seq)
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ringsim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sim", help="run a job, write data/shards/<job_id>.h5")
    p.add_argument("--job", required=True)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--frames", help="a:b frame range")
    p.add_argument("--out-dir")
    p.set_defaults(fn=cmd_sim)
    p = sub.add_parser("profile", help="plot profiles + differential for one sequence")
    p.add_argument("--shard", required=True)
    p.add_argument("--seq")
    p.add_argument("--out")
    p.add_argument("--temp-c", type=float)
    p.set_defaults(fn=cmd_profile)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
