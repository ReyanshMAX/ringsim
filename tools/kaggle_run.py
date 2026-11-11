#!/usr/bin/env python3
"""Run a ringsim job on a Kaggle T4 and bring the result back. Stdlib only.

Agents call this; the user never has to touch Kaggle. See docs/INFRA.md.

  python tools/kaggle_run.py doctor                    check CLI, auth, python, git
  python tools/kaggle_run.py selftest                  push a smoke kernel (T4, versions, tiny k-Wave sim)
  python tools/kaggle_run.py run jobs/p1_free.yaml     build wheel -> ringsim-code dataset -> push kernel
                                                       -> wait -> pull to data/kaggle/<job_id>/
  python tools/kaggle_run.py status <job_id>           one status poll
  python tools/kaggle_run.py pull <job_id>             pull output again
  add --dry-run to any command: build and render everything, call kaggle for nothing

Exit codes: 0 ok, 1 failed, 2 usage/config, 3 wrong GPU, 4 timeout, 5 another run holds the lock.
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
LOCK_STALE_S = 20 * 60
DRY = False

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
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        die("`kaggle` CLI not found. pip install kaggle, then `kaggle auth login`.", 2)
    if check and r.returncode != 0:
        die(f"kaggle {' '.join(args[:2])} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}", 1)
    return r


def detect_username(cfg: dict) -> str:
    u = cfg.get("username") or os.environ.get("KAGGLE_USERNAME", "")
    if u:
        return u
    if DRY:
        return "dry-run-user"
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

    def __enter__(self):
        if DRY:
            return self
        deadline = time.time() + 10 * 60
        while True:
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.time()}".encode())
                os.close(fd)
                self.held = True
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

    def release(self):
        if self.held:
            self.held = False
            LOCK.unlink(missing_ok=True)

    def __exit__(self, *a):
        self.release()


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
import base64, glob, json, os, subprocess, sys, time
EXPECT = "__BUILD_ID__"          # "" in selftest
MODE = "__MODE__"                # "job" | "selftest"
CMD = base64.b64decode("__CMD_B64__").decode()
JOB_YAML = base64.b64decode("__JOB_B64__").decode()
K = os.environ.get("KAGGLE_ROOT", "/kaggle"); W = K + "/working"; OUT = W + "/out"; os.makedirs(OUT, exist_ok=True)
meta = {"mode": MODE, "build_id": EXPECT, "python": sys.version.split()[0], "t0": time.time()}

def sh(c, **kw):
    print("$", c, flush=True)
    return subprocess.run(c, shell=True, **kw)

def finish(rc, why=""):
    meta.update(rc=rc, why=why, wall_s=round(time.time() - meta["t0"], 1))
    json.dump(meta, open(W + "/run_meta.json", "w"), indent=1)
    print("RUN_META", json.dumps(meta), flush=True)
    sys.exit(rc)

print("python", sys.version, flush=True)
gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
meta["gpu"] = gpu; print("GPU:", gpu or "none", flush=True)
sh("nvidia-smi")
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
    if sh(f"{sys.executable} -m pip install -q '{whl[0]}'").returncode:
        finish(5, "PIP_INSTALL_FAILED")
    sh(f"{sys.executable} -m pip install -q --force-reinstall --no-deps '{whl[0]}'")   # same version string, new code
    for p in glob.glob(K + "/input/*") + glob.glob(K + "/input/*/*"):
        if os.path.basename(p.rstrip("/")) == "ringsim-assets":
            os.environ["RINGSIM_DATA"] = p; meta["assets"] = p
    os.environ["RINGSIM_BUILD_ID"] = EXPECT      # shards record this; kernels have no git
    open(W + "/job.yaml", "w").write(JOB_YAML)
else:
    sh(f"{sys.executable} -m pip install -q k-wave-python==0.6.2")
sh(f"{sys.executable} -m pip show k-wave-python | head -3")
meta["kwave"] = subprocess.run(f"{sys.executable} -m pip show k-wave-python", shell=True, capture_output=True, text=True).stdout

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
        g = kWaveGrid(Vector([N, N, N]), Vector([dx] * 3)); g.setTime(nt, dt)
        med = kWaveMedium(sound_speed=c, density=1.2)
        src = kSource(); src.p_mask = np.zeros((N, N, N), bool); src.p_mask[N // 2, N // 2, 10] = True
        t = np.arange(nt) * dt - 125e-6
        src.p = (np.exp(-t**2 / (2 * 31.2e-6**2)) * np.sin(2 * np.pi * 22e3 * t))[None, :].astype(np.float32)
        src.p_mode = "additive"
        sen = kSensor(np.zeros((N, N, N), bool)); sen.mask[N // 2, N // 2, 20] = True; sen.record = ["p"]
        res = {}
        for gpu_flag in (False, True):
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


def render_kernel(build: str, mode: str, cmd: str, job_text: str) -> str:
    b64 = lambda s: base64.b64encode(s.encode()).decode()
    return (KERNEL_TEMPLATE.replace("__BUILD_ID__", build).replace("__MODE__", mode)
            .replace("__CMD_B64__", b64(cmd)).replace("__JOB_B64__", b64(job_text)))


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
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
            return st
        if time.time() - t0 > timeout_min * 60:
            log("timeout; kernel left running. Pull later with: pull <job_id>")
            return "TIMEOUT"
        time.sleep(cfg["poll_s"])


def pull(ref: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    kg("kernels", "output", ref, "-p", str(out))
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
    if st != "COMPLETE" or meta.get("rc") not in (0, None):
        rc = meta.get("rc")
        sys.exit(3 if rc == 3 else 1)


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


def cmd_run(cfg, a):
    job = Path(a.job)
    if not job.is_absolute():
        job = Path.cwd() / job
    if not job.exists():
        die(f"job file not found: {job}", 2)
    job_id = job.stem
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
    elif a.c == "run":
        cmd_run(cfg, a)
    elif a.c in ("status", "pull"):
        user = detect_username(cfg)
        slug = "ringsim-selftest" if a.job_id == "selftest" else f"ringsim-{slugify(a.job_id)}"
        if a.c == "status":
            print(parse_status(kg("kernels", "status", f"{user}/{slug}", check=False).stdout))
        else:
            meta = pull(f"{user}/{slug}", stage_dir(a.job_id) / "out")
            finish_report("COMPLETE", meta, stage_dir(a.job_id) / "out")


if __name__ == "__main__":
    main()
