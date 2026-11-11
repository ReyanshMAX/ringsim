# HANDOFF — seat reports, newest round first

The lead (seat 1) owns this file's header and lane table. Every seat writes ONLY its own section of the current
round, edited from main's copy (`git show main:docs/HANDOFF.md`). If `scripts/private-commit.sh` refuses this file,
re-read main's copy, put back only your own section and commit again; never hand-merge another seat's section.
STATUS.md and the Obsidian vault are written by the lead only, from these sections.

## Lanes

A lane is a set of files one seat owns and nobody else edits. Need a change in someone else's file: post to its owner.

| Seat | Lane | Owns | Keeps current |
|---|---|---|---|
| 1 LEAD | plan, contracts, review | STATUS.md, DECISIONS.md, OPEN_QUESTIONS.md, BUILD.md checkboxes, CLAUDE.md, docs/ARCHITECTURE.md, docs/OBSIDIAN.md, docs/HANDOFF.md (header + lane table), scripts/*.sh, scripts/obsidian_sync.py, .gitignore, .gitattributes, nextsessions/NEXT-PROMPTS.md, the Obsidian vault. No feature code. | ARCHITECTURE |
| 2 SIM | devices, geometry, solver, CLI, Kaggle | pyproject.toml, src/ringsim/__init__.py, src/ringsim/cli.py, src/ringsim/devices/** (incl. yaml configs), src/ringsim/geometry/**, src/ringsim/solver/**, jobs/*.yaml, tools/kaggle_run.py, tools/test_kaggle_run.py, kaggle/config.toml, tests/test_devices*.py, tests/test_geometry*.py, tests/test_solver*.py, tests/test_cli*.py | DEVICES, GEOMETRY, SOLVER, INFRA |
| 3 PROC | ringsim.proc (shared with the hardware) | src/ringsim/proc/**, tests/test_proc_*.py (incl. test_proc_isolation.py), tests/test_temperature_scaling.py (slow; seat 2 runs it on Kaggle) | PROCESSING |
| 4 DATASET AND VALIDATION | shards, reader, free-field check, test harness | src/ringsim/dataset/** (incl. jobspec.py, Q-030), src/ringsim/validation/**, tests/conftest.py, every test file not owned above | DATASET, VALIDATION |
| 5 SPECIALIST | OFF this round | Phase 2 candidate: capture-app/**, .github/workflows/ios.yml | CAPTURE_APP |

Bootstrap: seat 2's skeleton (9f682af) created every subpackage's empty `__init__.py`. From that sha on,
`proc/__init__.py` is seat 3's and `dataset/__init__.py`, `validation/__init__.py` are seat 4's.

One owner each: Kaggle runs and the `ringsim-code` upload = seat 2 · full `pytest -q` before round end = seat 4 ·
BUILD.md boxes, STATUS.md, vault = seat 1 · push = the user · a global install = the seat that needs it, after the
user's yes in that seat's own chat.

---

# Round 1 — Phase 1 (2026-11-11)

Goal: every Phase 1 acceptance criterion in BUILD.md checked, each with the command that proved it.
Room: `#ringsim`. Main at round start: ea6dfc5.

## Seat 1 LEAD

(written at round end)

## Seat 2 SIM

## Seat 3 PROC

## Seat 4 DATASET AND VALIDATION

## Seat 5 SPECIALIST (off)
