"""Mirror the ring-sim spec suite into the Obsidian vault (docs/OBSIDIAN.md).

Stdlib only. Overwrites generated mirrors; never touches overview/log/decisions.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROJECT_DIR = Path("03-competitions") / "isef-ring-pose"
ROOT_DOCS = ["CLAUDE.md", "BUILD.md", "DECISIONS.md", "OPEN_QUESTIONS.md"]
FRONTMATTER = "---\ntype: ref\nstatus: active\ntags: [isef, sim-spec]\n---\n"


def git_hash() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def mirror_name(rel: Path) -> str:
    return "isef-sim-" + rel.stem.lower().replace("_", "-") + ".md"


def render(src: Path, rel: Path, ref: str) -> str:
    body = src.read_text(encoding="utf-8")
    note = f"> Generated mirror of ring-sim/{rel.as_posix()} @ {ref}. Edit the repo, not this note.\n\n"
    return FRONTMATTER + note + body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default=os.environ.get("OBSIDIAN_VAULT"))
    args = ap.parse_args()
    if not args.vault or not Path(args.vault).is_dir():
        print("OBSIDIAN_VAULT unset or not a directory", file=sys.stderr)
        return 1
    proj = Path(args.vault) / PROJECT_DIR
    spec = proj / "sim-spec"
    spec.mkdir(parents=True, exist_ok=True)
    ref = git_hash()

    jobs: list[tuple[Path, Path]] = [(Path("STATUS.md"), proj / "isef-sim-status.md")]
    for name in ROOT_DOCS:
        jobs.append((Path(name), spec / mirror_name(Path(name))))
    for p in sorted((REPO / "docs").glob("*.md")):
        rel = p.relative_to(REPO)
        jobs.append((rel, spec / mirror_name(rel)))

    for rel, dst in jobs:
        src = REPO / rel
        if not src.exists():
            continue
        dst.write_text(render(src, rel, ref), encoding="utf-8", newline="\n")
        print(f"{rel.as_posix()} -> {dst.relative_to(args.vault).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
