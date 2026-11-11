"""CLI skeleton: every ARCHITECTURE.md subcommand exists; non-Phase-1 ones name their phase."""
import subprocess
import sys

import pytest

from ringsim.cli import main

SUBCOMMANDS = ["sim", "profile", "export-tx", "real", "validate", "converge"]


def test_help_lists_every_subcommand():
    out = subprocess.run([sys.executable, "-m", "ringsim.cli", "--help"],
                         capture_output=True, text=True, check=True).stdout
    for name in SUBCOMMANDS:
        assert name in out


@pytest.mark.parametrize("argv, phase", [
    (["validate", "plate", "--sim", "x.h5", "--real", "d", "--out", "r"], "Phase 2"),
    (["validate", "hand", "--sim", "x.h5", "--real", "d", "--out", "r"], "Phase 4"),
    (["converge", "--job", "jobs/converge.yaml", "--dx", "2.0,1.5,1.0"], "Phase 3"),
])
def test_later_phase_stubs_name_their_phase(argv, phase):
    with pytest.raises(NotImplementedError, match=phase):
        main(argv)
