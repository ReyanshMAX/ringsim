# OBSIDIAN.md — writing the project's work log and docs into the Obsidian vault

## Overview

The Obsidian vault is the human-facing record of this project: project overview, work
log, project-level decisions, and a read-only mirror of this spec suite. The repo stays
the source of truth for specs and code; the vault is where progress is narrated and
browsed. Every Claude Code session that changes anything ends by updating the vault.

## Non-goals

- The vault never becomes a second source of truth for specs. Mirrors are regenerated
  from the repo by script, never edited by hand in the vault.
- No vault git commits; the vault's own git is the user's.
- No writes outside `03-competitions/isef-ring-pose/` and `_attachments/` without asking.
- No code, data, shards, or captures in the vault. Small PNG figures only (≤1 MB each).

## Location

```
OBSIDIAN_VAULT = C:\Users\reyan\Desktop\knowledge-base          (env var; required for sync)
<vault>/03-competitions/isef-ring-pose/
  isef-overview.md            project overview — stable what/why/status, rewritten in place
  isef-log.md                 work log — append-only, newest at bottom, `## YYYY-MM-DD`
  isef-decisions.md           project-level decisions (chosen / rejected / why)
  isef-sim-status.md          mirror of STATUS.md (generated)
  sim-spec/isef-sim-*.md      mirrors of every other spec doc (generated)
  refs/                       reference notes (paper notes etc.)
<vault>/_attachments/isef-*.{pdf,png}    binaries, flat, prefixed `isef-`
```

## Vault rules that apply (from the vault's root CLAUDE.md — read it if unsure)

- Every note has frontmatter: `type` ∈ {project, competition, class, log, note, ref,
  agent}; `status` ∈ {active, paused, done, someday}; `tags: []`. Enumerated values only.
- Filenames are unique across the vault and lowercase-hyphen. Prefix every new note
  with `isef-`. Never create a file named `CLAUDE.md`, `STATUS.md`, etc. in the vault.
- `[[wikilinks]]` by bare filename.
- Never delete a vault note; move it. Never write into `09-archive/`.
- Never invent content to fill a note. No credentials, Kaggle keys, or tokens.
- Do not write a zero-length file (hangs the Obsidian Local REST API MCP server); write
  at least a newline.

## When to write

| Event | Write |
|---|---|
| End of any session that changed code, specs, data, or measurements | append to `isef-log.md`; run sync |
| Acceptance criterion checked / phase completed | log entry naming it; update the Status section of `isef-overview.md` |
| New entry in DECISIONS.md that changes project direction (method, scope, validation target, hardware) | row in `isef-decisions.md` linking the D-ID |
| Measurement added to STATUS.md | log entry with the number and its source command |
| Validation report or key plot produced | copy PNG to `_attachments/isef-<name>.png`, embed `![[isef-<name>.png]]` in the log entry |
| Something believed turned out wrong | log entry under **Disproven**, stating the old belief and the evidence |

Implementation-level decisions (grid constants, file formats) stay only in DECISIONS.md
and reach the vault through the mirror.

## Log entry format

Append under today's heading; create the heading if missing. Never edit past entries.

```markdown
## 2026-10-14

**Session:** Phase 1 — free-field pipeline
- Done: `ringsim.proc` + tests; `pytest -q` green (31 passed)
- Measured: wall_s/frame CPU = 4.2 s (p1_free, dx 2 mm) — `ringsim sim --job jobs/p1_free.yaml`
- Decided: none
- Disproven: none
- Blocked: Q-010 still open
- Next: `ringsim kaggle push --job jobs/p1_free.yaml`
```

Every number in a log entry comes from a command output or a file in the repo, with the
source named. Omit a line rather than writing a guess.

## Mirror sync

`scripts/obsidian_sync.py` (stdlib only, so it runs without the package installed):

```
python scripts/obsidian_sync.py            # uses $OBSIDIAN_VAULT
python scripts/obsidian_sync.py --vault "C:\Users\reyan\Desktop\knowledge-base"
```

Mapping: `STATUS.md → isef-sim-status.md` (folder root); every other spec doc
`X.md` or `docs/X.md → sim-spec/isef-sim-<x>.md`. Each mirror gets frontmatter
`type: ref, status: active, tags: [isef, sim-spec]` and a first line
`> Generated mirror of ring-sim/<path> @ <git short hash>. Edit the repo, not this note.`
The script overwrites mirrors only; it never touches overview, log, or decisions.

## Notes

- If `OBSIDIAN_VAULT` is unset or the path does not exist, the sync script exits 1 with
  a message. Log that in STATUS.md as blocked; do not write the vault anywhere else.
- If Claude Code is running somewhere without the vault (e.g. a Kaggle kernel), skip
  vault writes; the next local session catches up the log from git history.
