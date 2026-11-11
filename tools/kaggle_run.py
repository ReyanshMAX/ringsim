#!/usr/bin/env python3
"""Run a ringsim job on a Kaggle T4 and bring the result back. Stdlib only.

Agents call this; the user never has to touch Kaggle. See docs/INFRA.md.

  python tools/kaggle_run.py doctor                    check CLI, auth, python, git
  python tools/kaggle_run.py selftest                  push a smoke kernel (T4, versions, tiny k-Wave sim)
  python tools/kaggle_run.py bench                     time ring-sized k-Wave grids on T4 (dx 2/1.5/1 mm, 1 and 2 GPUs)
  python tools/kaggle_run.py run jobs/p1_free.yaml     build wheel -> ringsim-code dataset -> push kernel
                                                       -> wait -> pull to data/kaggle/<job_id>/
  python tools/kaggle_run.py status <job_id>           one status poll
  python tools/kaggle_run.py pull <job_id>             pull output again
  add --dry-run to any command: build and render everything, call kaggle for nothing

Exit codes: 0 ok, 1 failed (incl. no run_meta.json), 2 usage/config (incl. job_id != file name),
3 wrong GPU, 4 timeout or stale/missing code dataset, 5 another run holds the lock.
"""
from __future__ import annotations

import argparse, base64, csv, hashlib, io, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # python < 3.11
    tomllib = None

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = os.environ.get("KAGGLE_BIN", "kaggle")
LOCK = Path(tempfile.gettempdir()) / "ringsim-kaggle.lock"
LOCK_STALE_S = 20 * 60          # stale = not refreshed for this long (the holder refreshes on every poll)
DRY = False
ACTIVE_LOCK = None

DEFAULTS = {
    "username": "",                       # empty -> KAGGLE_USERNAME env -> detected from `kaggle ... --mine`
    "code_dataset": "ringsim-code",
    "assets_dataset": "ringsim-assets",
    "use_assets": False,                  # flip to true once the private assets dataset exists (MANO, poses)
    "accelerator": "NvidiaTeslaT4",
    "cmd": "ringsim sim --job {job} --gpu --out-dir {out}",
    "poll_s": 30,
    "timeout_min": 120,
}


def die(msg: str, code: int = 1):
    print(f"kaggle_run: {msg}", file=sys.stderr)
    sys.exit(code)


def log(msg: str):
    print(f"[kaggle_run] {msg}", flush=True)


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    p = ROOT / "kaggle" / "config.toml"
    if p.exists():
        if tomllib is None:
            die("kaggle/config.toml needs Python >= 3.11 (tomllib)", 2)
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        for section in data.values():
            if isinstance(section, dict):
                cfg.update(section)
    return cfg


# ---------------------------------------------------------------- kaggle CLI
def kg(*args: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    cmd = [KAGGLE, *args]
    if DRY:
        log("DRY: " + " ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    if not quiet:
        log("$ " + " ".join(cmd))
    exe = shutil.which(KAGGLE) or KAGGLE          # resolves kaggle.exe / kaggle.cmd on Windows
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))   # Windows cp1252 breaks log download
    except OSError:
        die("`kaggle` CLI not found or not runnable. pip install kaggle, then `kaggle auth login`.", 2)
    if check and r.returncode != 0:
        die(f"kaggle {' '.join(args[:2])} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}", 1)
    return r


def detect_username(cfg: dict) -> str:
    u = cfg.get("username") or os.environ.get("KAGGLE_USERNAME", "")
    if u:
        return u
    if DRY:
        return "dry-run-user"
    r = kg("config", "view", check=False, quiet=True)        # works on a fresh account with nothing published
    if r.returncode == 0:
        m = re.search(r"^\W*username:\s*(\S+)", r.stdout, re.M)
        if m and m.group(1).lower() not in ("none", "null"):
            return m.group(1)
    for args in (("kernels", "list", "--mine", "--csv"), ("datasets", "list", "--mine", "--csv")):
        r = kg(*args, check=False, quiet=True)
        if r.returncode == 0:
            for row in csv.DictReader(io.StringIO(r.stdout)):
                ref = row.get("ref", "")
                if "/" in ref:
                    return ref.split("/")[0].replace("https://www.kaggle.com/", "")
    die("cannot determine Kaggle username. Set [kaggle] username in kaggle/config.toml "
        "or run `kaggle auth login`, then retry.", 2)


# ---------------------------------------------------------------- lock
class Lock:
    """One push at a time across seats/worktrees: ringsim-code holds one build, and a kernel mounts
    whatever version is latest when it starts."""

    held = False
    token = ""

    def __enter__(self):
        global ACTIVE_LOCK
        if DRY:
            return self
        deadline = time.time() + 10 * 60
        while True:
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                self.token = f"{os.getpid()}-{time.time_ns()}"
                os.write(fd, self.token.encode())
                os.close(fd)
                self.held = True
                ACTIVE_LOCK = self
                return self
            except FileExistsError:
                try:
                    age = time.time() - LOCK.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > LOCK_STALE_S:
                    LOCK.unlink(missing_ok=True)
                    continue
                if time.time() > deadline:
                    die(f"another kaggle_run holds {LOCK} (age {age:.0f}s)", 5)
                time.sleep(5)

    def _mine(self) -> bool:
        try:
            return LOCK.read_text().strip() == self.token
        except OSError:
            return False

    def refresh(self):
        """Heartbeat: keeps the lock from looking stale during long dataset/queue waits."""
        if self.held and self._mine():
            try:
                os.utime(LOCK)
            except OSError:
                pass

    def release(self):
        global ACTIVE_LOCK
        if self.held:
            self.held = False
            if self._mine():                 # never delete a lock another run took over
                LOCK.unlink(missing_ok=True)
        if ACTIVE_LOCK is self:
            ACTIVE_LOCK = None

    def __exit__(self, *a):
        self.release()


def heartbeat():
    if ACTIVE_LOCK is not None:
        ACTIVE_LOCK.refresh()


# ---------------------------------------------------------------- code identity + wheel
def code_files() -> list[Path]:
    files = sorted(p for p in (ROOT / "src").rglob("*") if p.is_file() and "__pycache__" not in p.parts
                   and not p.name.endswith((".pyc", ".pyo")))
    py = ROOT / "pyproject.toml"
    return files + ([py] if py.exists() else [])


def build_id() -> str:
    """Hash of everything that goes into the wheel. Works with uncommitted changes."""
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def git_info() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain", "--", "src", "pyproject.toml"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
        return f"{h}{'+dirty' if d else ''}"
    except Exception:
        return "nogit"


def build_wheel(dest: Path) -> Path:
    if not (ROOT / "pyproject.toml").exists() or not (ROOT / "src").is_dir():
        die("no pyproject.toml/src yet: nothing to ship. (selftest does not need them.)", 2)
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.whl"):
        old.unlink()
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "pkg"
        shutil.copytree(ROOT / "src", stage / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
        shutil.copy2(ROOT / "pyproject.toml", stage / "pyproject.toml")
        for extra in ("README.md", "LICENSE"):
            if (ROOT / extra).exists():
                shutil.copy2(ROOT / extra, stage / extra)
        r = subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-q", "-w", str(dest)],
                           cwd=stage, capture_output=True, text=True)
        if r.returncode != 0:
            die(f"wheel build failed:\n{r.stdout}\n{r.stderr}", 1)
    wheels = list(dest.glob("*.whl"))
    if len(wheels) != 1:
        die(f"expected exactly one wheel in {dest}, got {wheels}", 1)
    return wheels[0]


# ---------------------------------------------------------------- kernel script
KERNEL_TEMPLATE = r'''
import base64, glob, json, os, shutil, subprocess, sys, time
EXPECT = "__BUILD_ID__"          # "" in selftest
MODE = "__MODE__"                # "job" | "selftest"
CMD = base64.b64decode("__CMD_B64__").decode()
JOB_YAML = base64.b64decode("__JOB_B64__").decode()
K = os.environ.get("KAGGLE_ROOT", "/kaggle"); W = K + "/working"; OUT = W + "/out"; os.makedirs(OUT, exist_ok=True)
meta = {"mode": MODE, "build_id": EXPECT, "python": sys.version.split()[0], "t0": time.time()}

def sh(c, **kw):                 # the job command only: a user-supplied command line
    print("$", c, flush=True)
    return subprocess.run(c, shell=True, **kw)

def run(argv, **kw):             # everything else: argument lists, no shell, same on Linux and Windows
    print("$", " ".join(argv), flush=True)
    return subprocess.run([shutil.which(argv[0]) or argv[0], *argv[1:]], **kw)

PY = [sys.executable, "-m", "pip"]

def finish(rc, why=""):
    meta.update(rc=rc, why=why, wall_s=round(time.time() - meta["t0"], 1))
    json.dump(meta, open(W + "/run_meta.json", "w"), indent=1)
    print("RUN_META", json.dumps(meta), flush=True)
    sys.exit(rc)

print("python", sys.version, flush=True)
smi = shutil.which("nvidia-smi")
gpu = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() if smi else ""
meta["gpu"] = gpu; print("GPU:", gpu or "none", flush=True)
if smi:
    run(["nvidia-smi"])
if "T4" not in gpu:
    finish(3, "WRONG_GPU: need T4 (P100 cannot run the k-Wave CUDA binary, D-016)")

if MODE == "job":
    ids = glob.glob(K + "/input/**/BUILD_ID", recursive=True)
    if not ids:
        finish(4, "NO_CODE_DATASET: ringsim-code is not mounted")
    got = open(ids[0]).read().strip(); meta["got_build_id"] = got
    if got != EXPECT:
        finish(4, f"BUILD_ID_MISMATCH: kernel mounted {got}, expected {EXPECT} (stale dataset version)")
    whl = glob.glob(os.path.dirname(ids[0]) + "/*.whl")
    if len(whl) != 1:
        finish(4, f"WHEEL_COUNT: {whl}")
    if run(PY + ["install", "-q", whl[0] + "[sim]"]).returncode:        # [sim]: everything beyond numpy/scipy
        finish(1, "PIP_INSTALL_FAILED")
    run(PY + ["install", "-q", "--force-reinstall", "--no-deps", whl[0]])   # same version string, new code
    for p in glob.glob(K + "/input/*") + glob.glob(K + "/input/*/*"):
        if os.path.basename(p.rstrip("/")) == "ringsim-assets":
            os.environ["RINGSIM_DATA"] = p; meta["assets"] = p
    os.environ["RINGSIM_BUILD_ID"] = EXPECT      # shards record this; kernels have no git
    open(W + "/job.yaml", "w").write(JOB_YAML)
else:
    run(PY + ["install", "-q", "k-wave-python==0.6.2"])
meta["kwave"] = "\n".join(subprocess.run(PY + ["show", "k-wave-python"], capture_output=True, text=True).stdout.splitlines()[:2])
print(meta["kwave"], flush=True)

if MODE == "bench":
    import re as _re
    open(W + "/bench_case.py", "w").write(base64.b64decode("__BENCH_B64__").decode())
    cases = json.loads(base64.b64decode("__CASES_B64__").decode())
    ngpu = len([l for l in gpu.splitlines() if l.strip()])
    def launch(case, dev, tag):
        tmp = W + "/tmp_" + tag; os.makedirs(tmp, exist_ok=True)   # k-wave-python names its HDF5 files by time
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(dev), TMPDIR=tmp)
        return subprocess.Popen([sys.executable, W + "/bench_case.py", json.dumps(case)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def collect(p):
        out, _ = p.communicate()
        print(out[-2500:], flush=True)
        res = {"error": out[-600:]}
        for line in reversed(out.splitlines()):
            if line.startswith("BENCH_RESULT "):
                res = json.loads(line[len("BENCH_RESULT "):]); break
        res["kwave_total_s"] = [float(x) for x in _re.findall(r"Total execution time:\s*([\d.]+)s", out)]
        res["kwave_stage_s"] = [float(x) for x in _re.findall(r"Elapsed time:\s*([\d.]+)s", out)]
        return res
    results = [collect(launch(c, 0, c["name"])) for c in cases]
    dual = None
    if ngpu >= 2:
        c2 = [c for c in cases if c["name"] == "ring_dx2.0"][0]
        t1 = time.time()
        rs = [collect(p) for p in [launch(c2, d, "dual%d" % d) for d in (0, 1)]]
        dual = {"case": c2["name"], "wall_s_total": round(time.time() - t1, 2), "per_gpu": rs}
    meta["bench"] = {"ngpu": ngpu, "cases": results, "dual": dual}
    ok = all("error" not in r and r.get("finite") for r in results)
    finish(0 if ok else 1, "" if ok else "BENCH_FAILED")

if MODE == "selftest":
    try:
        import numpy as np
        from kwave.data import Vector
        from kwave.kgrid import kWaveGrid
        from kwave.kmedium import kWaveMedium
        from kwave.ksource import kSource
        from kwave.ksensor import kSensor
        from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
        from kwave.options.simulation_options import SimulationOptions
        from kwave.options.simulation_execution_options import SimulationExecutionOptions
        N, dx, c = 48, 2e-3, 343.0
        dt = 0.3 * dx / c; nt = 300
        def build():   # fresh objects per call: kspaceFirstOrder3D modifies grid/source/sensor in place
            g = kWaveGrid(Vector([N, N, N]), Vector([dx] * 3)); g.setTime(nt, dt)
            med = kWaveMedium(sound_speed=c, density=1.2)
            src = kSource(); src.p_mask = np.zeros((N, N, N), bool); src.p_mask[N // 2, N // 2, 10] = True
            t = np.arange(nt) * dt - 125e-6
            src.p = (np.exp(-t**2 / (2 * 31.2e-6**2)) * np.sin(2 * np.pi * 22e3 * t))[None, :].astype(np.float32)
            src.p_mode = "additive"
            sen = kSensor(np.zeros((N, N, N), bool)); sen.mask[N // 2, N // 2, 20] = True; sen.record = ["p"]
            return g, med, src, sen
        res = {}
        for gpu_flag in (False, True):
            g, med, src, sen = build()
            t1 = time.time()
            o = kspaceFirstOrder3D(kgrid=g, medium=med, source=src, sensor=sen,
                                   simulation_options=SimulationOptions(pml_inside=False, pml_size=10, data_cast="single", save_to_disk=True),
                                   execution_options=SimulationExecutionOptions(is_gpu_simulation=gpu_flag))
            p = np.asarray(o["p"])
            res["gpu" if gpu_flag else "cpu"] = {"shape": list(p.shape), "finite": bool(np.isfinite(p).all()),
                                                  "peak": float(np.abs(p).max()), "wall_s": round(time.time() - t1, 2)}
        meta["ksim"] = res
        ok = all(v["finite"] and v["peak"] > 0 for v in res.values())
        finish(0 if ok else 1, "" if ok else "KSIM_BAD_OUTPUT")
    except Exception as e:
        import traceback; traceback.print_exc()
        finish(1, f"KSIM_EXCEPTION: {e!r}")

rc = sh(CMD.replace("{job}", W + "/job.yaml").replace("{out}", OUT)).returncode
finish(rc, "" if rc == 0 else "CMD_FAILED")
'''


BENCH_CASE = r'''
import json, sys, time, warnings
import numpy as np
from scipy.ndimage import gaussian_filter
warnings.filterwarnings("ignore", category=FutureWarning)
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
c = json.loads(sys.argv[1]); N = c["dims"]; dx = c["dx"]; nt = c["nt"]; dt = c["dt"]
rho = np.full(N, 1.2, np.float32)                      # heterogeneous like a real snapshot: one smoothed
rho[N[0] // 3: N[0] // 2, N[1] // 4: 3 * N[1] // 4, N[2] // 4: N[2] // 2] = 1000.0   # solid block (D-036), kept
                                                       # off the grid faces: a solid touching them goes NaN (Q-024)
rho = gaussian_filter(rho, 0.5).astype(np.float32)
i0 = (N[0] * 2 // 3, N[1] // 2, N[2] // 2)
walls = []
for rep in range(c["reps"]):
    g = kWaveGrid(Vector(list(N)), Vector([dx] * 3)); g.setTime(nt, dt)      # fresh objects every call
    med = kWaveMedium(sound_speed=343.0, density=rho)
    src = kSource(); src.p_mask = np.zeros(N, bool); src.p_mask[i0] = True
    t = np.arange(nt) * dt - 125e-6
    src.p = (np.exp(-t**2 / (2 * 31.2e-6**2)) * np.sin(2 * np.pi * 22e3 * t))[None, :].astype(np.float32)
    src.p_mode = "additive"
    sen = kSensor(np.zeros(N, bool)); sen.mask[i0[0], i0[1] + 3, i0[2]] = True; sen.record = ["p"]
    t1 = time.time()
    o = kspaceFirstOrder3D(kgrid=g, medium=med, source=src, sensor=sen,
                           simulation_options=SimulationOptions(pml_inside=False, pml_size=10, data_cast="single", save_to_disk=True),
                           execution_options=SimulationExecutionOptions(is_gpu_simulation=True))
    walls.append(round(time.time() - t1, 3))
    p = np.asarray(o["p"])
print("BENCH_RESULT " + json.dumps(dict(c, wall_s=walls, finite=bool(np.isfinite(p).all()), peak=float(np.abs(p).max()))), flush=True)
'''

# Ring-sized benchmark (BUILD.md Phase 1, Q-007). Domain = ring-frame hand bounding box estimate
# (~200 x 110 x 70 mm) + 10 mm air each side; replace with measured scene sizes once MANO scenes exist.
BENCH_RING_MM = (220.0, 130.0, 90.0)


def fast_dims(n: int, pml: int = 10) -> int:
    m = n
    while True:
        x = m + 2 * pml
        for p in (2, 3, 5, 7):
            while x % p == 0:
                x //= p
        if x == 1:
            return m
        m += 1


def bench_cases() -> list[dict]:
    import math
    fs, n_bins, d_direct, c_sim, cfl = 50000, 54, 5.4e-3, 343.0, 0.3       # ring.yaml signal (D-029)
    c_max = 331.3 * math.sqrt(1 + 35 / 273.15)
    sigma = math.sqrt(2 * math.log(2)) / (2 * math.pi * 6000.0)
    t_end = 8 * sigma + d_direct / c_sim + (c_max / c_sim) * (n_bins / fs) + 0.15e-3   # SOLVER.md time_axis (D-044)
    cases = []
    for name, dx_mm, size in (("overhead_25", 2.0, (50.0, 50.0, 50.0)), ("ring_dx2.0", 2.0, BENCH_RING_MM),
                              ("ring_dx1.5", 1.5, BENCH_RING_MM), ("ring_dx1.0", 1.0, BENCH_RING_MM)):
        dx = dx_mm * 1e-3
        k = math.ceil(c_sim / (cfl * dx * fs)); dt = 1.0 / (k * fs)
        dims = [fast_dims(math.ceil(L / dx_mm)) for L in size]
        cases.append({"name": name, "dx": dx, "dims": dims, "dt": dt, "nt": math.ceil(t_end / dt), "k": k, "reps": 2,
                      "cells_with_pml": math.prod(d + 20 for d in dims)})
    return cases


def render_kernel(build: str, mode: str, cmd: str, job_text: str) -> str:
    b64 = lambda s: base64.b64encode(s.encode()).decode()
    return (KERNEL_TEMPLATE.replace("__BUILD_ID__", build).replace("__MODE__", mode)
            .replace("__CMD_B64__", b64(cmd)).replace("__JOB_B64__", b64(job_text))
            .replace("__BENCH_B64__", b64(BENCH_CASE if mode == "bench" else ""))
            .replace("__CASES_B64__", b64(json.dumps(bench_cases() if mode == "bench" else []))))


def slugify(s: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")


# ---------------------------------------------------------------- commands
def stage_dir(job_id: str) -> Path:
    d = ROOT / "data" / "kaggle" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_status(text: str) -> str:
    m = re.search(r"(?:KernelWorkerStatus\.)?\b(COMPLETE|ERROR|RUNNING|QUEUED|CANCEL_ACKNOWLEDGED|CANCEL_REQUESTED|NEW_SCRIPT)\b", text, re.I)
    return m.group(1).upper() if m else "UNKNOWN"


def upload_code(cfg: dict, user: str, bid: str, work: Path):
    pkg = work / "code"
    wheel = build_wheel(pkg)
    (pkg / "BUILD_ID").write_text(bid + "\n")
    ref = f"{user}/{cfg['code_dataset']}"
    (pkg / "dataset-metadata.json").write_text(json.dumps(
        {"title": cfg["code_dataset"], "id": ref, "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    log(f"build {bid} ({git_info()}) wheel {wheel.name}")
    exists = kg("datasets", "status", ref, check=False, quiet=True)
    if DRY or exists.returncode != 0 or "404" in exists.stdout + exists.stderr or "not found" in (exists.stdout + exists.stderr).lower():
        kg("datasets", "create", "-p", str(pkg), "--dir-mode", "skip")
    else:
        kg("datasets", "version", "-p", str(pkg), "-m", bid, "--dir-mode", "skip")
    deadline = time.time() + 10 * 60
    while not DRY:
        s = kg("datasets", "status", ref, check=False, quiet=True)
        txt = (s.stdout + s.stderr).lower()
        if "ready" in txt:
            break
        if "error" in txt or "fail" in txt:
            die(f"dataset processing failed: {txt}", 1)
        if time.time() > deadline:
            die("dataset not ready after 10 min", 4)
        heartbeat()
        time.sleep(10)


def push_kernel(cfg: dict, user: str, slug: str, mode: str, script: str, work: Path, with_code: bool):
    k = work / "kernel"
    k.mkdir(parents=True, exist_ok=True)
    (k / "run_job.py").write_text(script, encoding="utf-8")
    sources = []
    if with_code:
        sources.append(f"{user}/{cfg['code_dataset']}")
        if cfg.get("use_assets"):
            sources.append(f"{user}/{cfg['assets_dataset']}")
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "run_job.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": sources, "competition_sources": [], "kernel_sources": []}, indent=1))
    kg("kernels", "push", "-p", str(k), "--accelerator", cfg["accelerator"])


def wait_kernel(cfg: dict, ref: str, lock: Lock | None, timeout_min: float) -> str:
    if DRY:
        return "COMPLETE"
    t0 = time.time()
    last = ""
    while True:
        s = kg("kernels", "status", ref, check=False, quiet=True)
        st = parse_status(s.stdout + s.stderr)
        if st != last:
            log(f"{ref}: {st} ({(time.time() - t0) / 60:.1f} min)")
            last = st
        if st in ("RUNNING", "COMPLETE", "ERROR") and lock:   # datasets are mounted; others may push now
            lock.release(); lock = None
        elif lock:
            lock.refresh()
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
            return st
        if time.time() - t0 > timeout_min * 60:
            log("timeout; kernel left running. Pull later with: pull <job_id>")
            return "TIMEOUT"
        time.sleep(cfg["poll_s"])


def pull(ref: str, out: Path) -> dict:
    shutil.rmtree(out, ignore_errors=True)       # runner scratch: never report a previous run's files
    out.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2, 3):                    # transient SSL/HTTP errors happen; the output stays on Kaggle
        r = kg("kernels", "output", ref, "-p", str(out), check=False)
        if r.returncode == 0:
            break
        if attempt == 3:
            die(f"kaggle kernels output failed 3 times:\n{r.stdout}\n{r.stderr}\nRetry later with: pull <job_id>", 1)
        log(f"output download failed (attempt {attempt}); retrying in {15 * attempt} s")
        if not DRY:
            time.sleep(15 * attempt)
    meta = {}
    mp = out / "run_meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    shards = ROOT / "data" / "shards"
    for h5 in out.rglob("*.h5"):
        shards.mkdir(parents=True, exist_ok=True)
        shutil.copy2(h5, shards / h5.name)
        log(f"shard -> data/shards/{h5.name}")
    for lg in sorted(out.glob("*.log"))[:1]:
        lines = lg.read_text(errors="replace").splitlines()
        log(f"log tail ({lg.name}):\n" + "\n".join(lines[-25:]))
    return meta


def finish_report(st: str, meta: dict, out: Path):
    log(f"result: kernel={st} rc={meta.get('rc')} why={meta.get('why') or '-'} gpu={meta.get('gpu')} "
        f"wall_s={meta.get('wall_s')} build={meta.get('build_id') or '-'}  files in {out}")
    if DRY:
        return
    if st == "TIMEOUT":
        sys.exit(4)
    if not meta:
        log("NO_RUN_META: kernel output has no run_meta.json; treat the run as failed")
        sys.exit(1)
    if st != "COMPLETE" or meta.get("rc") != 0:
        rc = meta.get("rc")
        sys.exit(rc if rc in (3, 4) else 1)


def cmd_doctor(cfg):
    ok = True
    def row(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"{'ok  ' if good else 'FAIL'} {name} {detail}")
    row("python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    k = shutil.which(KAGGLE)
    row("kaggle CLI on PATH", k, k or "pip install kaggle")
    if k:
        r = kg("kernels", "list", "--mine", "--page-size", "1", check=False, quiet=True)
        row("kaggle auth works", r.returncode == 0, "" if r.returncode == 0 else "run `kaggle auth login`\n" + (r.stderr or r.stdout)[:300])
    row("git", shutil.which("git"))
    row("pyproject.toml + src/ (needed for `run`, not selftest)", (ROOT / "pyproject.toml").exists() and (ROOT / "src").is_dir(),
        "absent until Phase 1 creates them" if not (ROOT / "pyproject.toml").exists() else "")
    row("lock free", not LOCK.exists(), str(LOCK))
    sys.exit(0 if ok else 1)


def cmd_selftest(cfg, a):
    user = detect_username(cfg)
    slug = "ringsim-selftest"
    work = stage_dir("selftest")
    script = render_kernel("", "selftest", "true", "")
    with Lock() as lk:
        push_kernel(cfg, user, slug, "selftest", script, work, with_code=False)
        st = wait_kernel(cfg, f"{user}/{slug}", lk, a.timeout)
    meta = {} if DRY else pull(f"{user}/{slug}", work / "out")
    if not DRY:
        log("ksim: " + json.dumps(meta.get("ksim")))
    finish_report(st, meta, work / "out")


def bench_report(meta: dict) -> str:
    b = meta.get("bench") or {}
    rows = ["case         dims          cells+PML  steps  wall_s(min)  kwave_total(min)  snapshots/GPU-h"]
    for r in b.get("cases", []):
        if "error" in r:
            rows.append(f"{r.get('name', '?'):12s} ERROR {r['error'][-200:]!r}")
            continue
        w = min(r["wall_s"]); kt = min(r["kwave_total_s"]) if r["kwave_total_s"] else float("nan")
        rows.append(f"{r['name']:12s} {'x'.join(map(str, r['dims'])):13s} {r['cells_with_pml']:9d}  {r['nt']:5d}  "
                    f"{w:11.2f}  {kt:16.2f}  {3600 / w:15.0f}")
    d = b.get("dual")
    if d:
        n = sum(len(x.get("wall_s", [])) for x in d["per_gpu"])
        rows.append(f"two T4s at once, {d['case']}: {n} snapshots in {d['wall_s_total']:.1f} s "
                    f"-> {3600 * n / d['wall_s_total']:.0f} snapshots/hour for the pair (incl. process start-up)")
    return "\n".join(rows)


def cmd_bench(cfg, a):
    user = detect_username(cfg)
    slug = "ringsim-bench"
    work = stage_dir("bench")
    script = render_kernel("", "bench", "true", "")
    with Lock() as lk:
        push_kernel(cfg, user, slug, "bench", script, work, with_code=False)
        st = wait_kernel(cfg, f"{user}/{slug}", lk, a.timeout)
    meta = {} if DRY else pull(f"{user}/{slug}", work / "out")
    if not DRY:
        log("bench:\n" + bench_report(meta))
    finish_report(st, meta, work / "out")


def cmd_run(cfg, a):
    job = Path(a.job)
    if not job.is_absolute():
        job = Path.cwd() / job
    if not job.exists():
        die(f"job file not found: {job}", 2)
    m = re.search(r"^job_id:\s*['\"]?([^'\"\s#]+)", job.read_text(encoding="utf-8"), re.M)
    if not m:
        die(f"{job.name} has no job_id", 2)
    job_id = m.group(1)
    if job_id != job.stem:
        die(f"job_id '{job_id}' differs from the file name '{job.stem}'; they must match (kernel slug and shard name)", 2)
    user = detect_username(cfg)
    slug = f"ringsim-{slugify(job_id)}"
    work = stage_dir(job_id)
    bid = build_id()
    cmd = a.cmd or cfg["cmd"]          # {job} and {out} are filled in inside the kernel
    script = render_kernel(bid, "job", cmd, job.read_text(encoding="utf-8"))
    for attempt in (1, 2, 3):
        with Lock() as lk:
            upload_code(cfg, user, bid, work)
            push_kernel(cfg, user, slug, "job", script, work, with_code=True)
            if a.no_wait:
                log(f"pushed {user}/{slug}; poll with: status {job_id}"); lk.release(); return
            st = wait_kernel(cfg, f"{user}/{slug}", lk, a.timeout)
        meta = {} if DRY else pull(f"{user}/{slug}", work / "out")
        if str(meta.get("why", "")).startswith("BUILD_ID_MISMATCH") and attempt < 3:
            log("stale dataset mounted; retrying in 45 s"); time.sleep(45); continue
        break
    finish_report(st, meta, work / "out")


def main():
    global DRY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    sub = ap.add_subparsers(dest="c", required=True)
    sub.add_parser("doctor")
    s = sub.add_parser("selftest"); s.add_argument("--timeout", type=float, default=30)
    bp = sub.add_parser("bench"); bp.add_argument("--timeout", type=float, default=45)
    r = sub.add_parser("run"); r.add_argument("job"); r.add_argument("--cmd"); r.add_argument("--no-wait", action="store_true")
    r.add_argument("--timeout", type=float, default=None)
    t = sub.add_parser("status"); t.add_argument("job_id")
    p = sub.add_parser("pull"); p.add_argument("job_id")
    a = ap.parse_args()
    DRY = a.dry_run
    cfg = load_config()
    if getattr(a, "timeout", None) is None and a.c == "run":
        a.timeout = cfg["timeout_min"]
    if a.c == "doctor":
        cmd_doctor(cfg)
    elif a.c == "selftest":
        cmd_selftest(cfg, a)
    elif a.c == "bench":
        cmd_bench(cfg, a)
    elif a.c == "run":
        cmd_run(cfg, a)
    elif a.c in ("status", "pull"):
        user = detect_username(cfg)
        slug = f"ringsim-{a.job_id}" if a.job_id in ("selftest", "bench") else f"ringsim-{slugify(a.job_id)}"
        if a.c == "status":
            print(parse_status(kg("kernels", "status", f"{user}/{slug}", check=False).stdout))
        else:
            meta = pull(f"{user}/{slug}", stage_dir(a.job_id) / "out")
            finish_report("COMPLETE", meta, stage_dir(a.job_id) / "out")


if __name__ == "__main__":
    main()
