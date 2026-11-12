"""Protocol A analysis (VALIDATION.md, D-087): speaker isolation and side check, bottom-speaker response, hole-cover
mic identification, in-band level against the no-playback floor. Pure analysis of a session directory of captures;
the decisions are the thresholds VALIDATION.md gives (>= 20 dB grille drop AND < 3 dB earpiece drop; > 10 dB mic
drop; >= 20 dB above the floor). Nothing is chosen here beyond them.

Take stems: protA0_<left|right>_<open|ear|grille|rgrille>_t<k> (rgrille optional, D-087), protA1_sweep_t<k>, protA2_<mic2|mic1grp>_t<k>,
protA3_noplay_t<k>. Any other .wav stem raises.
"""
from __future__ import annotations

import os
import re

import numpy as np

from ringsim.validation.sweep import BAND_HZ, complex_response, noise_floor_response, phase_residual

RE0 = re.compile(r"^protA0_(left|right)_(open|ear|grille|rgrille)_t(\d+)$")
RE1 = re.compile(r"^protA1_sweep_t(\d+)$")
RE2 = re.compile(r"^protA2_(mic2|mic1grp)_t(\d+)$")
RE3 = re.compile(r"^protA3_noplay_t(\d+)$")

GRILLE_DROP_DB = 20.0       # step 0: bottom grille covered
EAR_DROP_DB = 3.0           # step 0: earpiece covered
MIC_DROP_DB = 10.0          # step 2
FLOOR_MARGIN_DB = 20.0      # step 3
NO_TAKES = {"status": "no takes"}


def _parse(session_dir: str) -> dict:
    g: dict = {"s0": {}, "s1": [], "s2": {}, "s3": []}
    for f in sorted(os.listdir(session_dir)):
        if not f.lower().endswith(".wav"):
            continue
        stem, path = os.path.splitext(f)[0], os.path.join(session_dir, f)
        if m := RE0.match(stem):
            g["s0"].setdefault(m.group(1), {}).setdefault(m.group(2), []).append(path)
        elif RE1.match(stem):
            g["s1"].append(path)
        elif m := RE2.match(stem):
            g["s2"].setdefault(m.group(1), []).append(path)
        elif RE3.match(stem):
            g["s3"].append(path)
        else:
            raise ValueError(f"unrecognised take {stem!r}: expected protA0_/protA1_/protA2_/protA3_ labels")
    return g


def level_db(sweep_wav: str, takes: list[str], band: tuple[float, float] = BAND_HZ) -> float:
    """Mean over `band` of 20 log10 |H| (sweep.response of these takes, aligned complex average)."""
    f, H = complex_response(sweep_wav, takes)
    m = (f >= band[0]) & (f <= band[1])
    return float(np.mean(20 * np.log10(np.maximum(np.abs(H[m]), 1e-30))))


def floor_db(sweep_wav: str, takes: list[str], band: tuple[float, float] = BAND_HZ) -> float:
    """Mean over `band` of the no-playback floor in dB (sweep.noise_floor_response, D-089)."""
    f, mag = noise_floor_response(sweep_wav, takes)
    m = (f >= band[0]) & (f <= band[1])
    return float(np.mean(mag[m]))


def _step0(sweep_wav: str, s0: dict) -> tuple[dict, str | None, str]:
    out = {}
    for ch in ("left", "right"):
        conds = s0.get(ch, {})
        if not all(c in conds for c in ("open", "ear", "grille")):
            continue
        lv = {c: level_db(sweep_wav, conds[c]) for c in ("open", "ear", "grille")}
        de, dg = lv["open"] - lv["ear"], lv["open"] - lv["grille"]
        out[ch] = {"level_db": lv, "drop_ear_db": de, "drop_grille_db": dg,
                   "bottom_only": bool(dg >= GRILLE_DROP_DB and de < EAR_DROP_DB)}
        if "rgrille" in conds:                                  # optional: only recorded when no channel qualified
            lv["rgrille"] = level_db(sweep_wav, conds["rgrille"])
            out[ch]["drop_rgrille_db"] = lv["open"] - lv["rgrille"]
            out[ch]["mirrored"] = bool(out[ch]["drop_rgrille_db"] >= GRILLE_DROP_DB and de < EAR_DROP_DB)
    if not out:
        return {"status": "no takes"}, None, "no takes"
    ok = [c for c, v in out.items() if v["bottom_only"]]
    if len(ok) == 1:
        return out, ok[0], "ok"
    if len(ok) == 2:
        return out, None, "ambiguous: both channels qualify"
    mir = [c for c, v in out.items() if v.get("mirrored")]
    if len(mir) == 1:
        return out, mir[0], ("mirrored: covering the RIGHT bottom group silences this channel; mirror x for the "
                             "bottom ports in iphone17pro.yaml and amend D-023")
    if len(mir) == 2:
        return out, None, "ambiguous: both channels mirrored"
    missing = [c for c in ("left", "right") if c not in out]
    if missing:
        return out, None, f"incomplete: no open/ear/grille set for {missing}"
    return out, None, (f"blocked: neither channel meets >= {GRILLE_DROP_DB:.0f} dB grille drop AND "
                       f"< {EAR_DROP_DB:.0f} dB earpiece drop")


def protocol_a(session_dir: str, sweep_wav: str) -> dict:
    g = _parse(session_dir)
    res: dict = {"session": session_dir, "band_hz": list(BAND_HZ),
                 "n_takes": {"step0": sum(len(v) for c in g["s0"].values() for v in c.values()),
                             "step1": len(g["s1"]), "step2": sum(len(v) for v in g["s2"].values()),
                             "step3": len(g["s3"])}}
    res["step0"], res["tx_channel"], res["step0_verdict"] = _step0(sweep_wav, g["s0"])

    lvl1 = None
    if g["s1"]:
        lvl1 = level_db(sweep_wav, g["s1"])
        fb, r = phase_residual(sweep_wav, g["s1"])
        res["step1"] = {"level_db": lvl1, "phase_residual_rms_rad": float(np.sqrt(np.mean(r ** 2))),
                        "phase_residual_max_rad": float(np.max(np.abs(r))), "band_hz": [float(fb[0]), float(fb[-1])]}
    else:
        res["step1"] = NO_TAKES

    if g["s2"] and lvl1 is not None:
        drops = {k: lvl1 - level_db(sweep_wav, v) for k, v in g["s2"].items()}
        hit = [k for k, d in drops.items() if d > MIC_DROP_DB]
        bottom = ("mic2" if hit == ["mic2"] else "mic1" if hit == ["mic1grp"] else "ambiguous")
        res["step2"] = {"level_open_db": lvl1, "drop_db": drops, "bottom_mic": bottom,
                        "rule": f"> {MIC_DROP_DB:.0f} dB drop"}
    else:
        res["step2"] = NO_TAKES if not g["s2"] else {"status": "needs protA1 takes as the open level"}

    if g["s3"] and lvl1 is not None:
        floor = floor_db(sweep_wav, g["s3"])
        res["step3"] = {"level_db": lvl1, "floor_db": floor, "margin_db": lvl1 - floor,
                        "pass": bool(lvl1 - floor >= FLOOR_MARGIN_DB), "rule": f">= {FLOOR_MARGIN_DB:.0f} dB"}
    else:
        res["step3"] = NO_TAKES if not g["s3"] else {"status": "needs protA1 takes for the level"}
    return res


def write_protocol_a_report(result: dict, path: str) -> None:
    L = ["# Protocol A report", "", f"Session: `{result['session']}`  ·  band {result['band_hz'][0]:.0f}–"
         f"{result['band_hz'][1]:.0f} Hz  ·  takes per step: {result['n_takes']}", "",
         "## Step 0: speaker isolation (D-032)", ""]
    s0 = result["step0"]
    if "status" in s0:
        L.append(f"No takes ({result['step0_verdict']}).")
    else:
        L += ["| channel | open dB | ear dB | grille dB | rgrille dB | ear drop | grille drop | rgrille drop | bottom only | mirrored |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for ch, v in s0.items():
            lv = v["level_db"]
            rg = f"{lv['rgrille']:.1f}" if "rgrille" in lv else "-"
            rd = f"{v['drop_rgrille_db']:.1f}" if "drop_rgrille_db" in v else "-"
            L.append(f"| {ch} | {lv['open']:.1f} | {lv['ear']:.1f} | {lv['grille']:.1f} | {rg} | {v['drop_ear_db']:.1f} | "
                     f"{v['drop_grille_db']:.1f} | {rd} | {v['bottom_only']} | {v.get('mirrored', '-')} |")
        L += ["", f"Verdict: **{result['step0_verdict']}**; txChannel = `{result['tx_channel']}`."]
    L += ["", "## Step 1: bottom-speaker response", ""]
    s1 = result["step1"]
    L.append("No takes." if "status" in s1 else
             f"Level {s1['level_db']:.1f} dB; phase residual after the best-fit line over "
             f"{s1['band_hz'][0]:.0f}–{s1['band_hz'][1]:.0f} Hz: rms {s1['phase_residual_rms_rad']:.3f} rad, "
             f"max {s1['phase_residual_max_rad']:.3f} rad.")
    L += ["", "## Step 2: hole-cover mic identification", ""]
    s2 = result["step2"]
    L.append(s2["status"] if "status" in s2 else
             f"Drops vs open ({s2['level_open_db']:.1f} dB): " + ", ".join(f"{k} {d:.1f} dB" for k, d in s2["drop_db"].items())
             + f". Rule {s2['rule']} → bottom mic: **{s2['bottom_mic']}**.")
    L += ["", "## Step 3: band against the no-playback floor", ""]
    s3 = result["step3"]
    L.append(s3["status"] if "status" in s3 else
             f"Level {s3['level_db']:.1f} dB, floor {s3['floor_db']:.1f} dB, margin {s3['margin_db']:.1f} dB; "
             f"rule {s3['rule']} → **{'pass' if s3['pass'] else 'FAIL'}**.")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
