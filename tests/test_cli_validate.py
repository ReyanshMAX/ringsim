"""`ringsim validate plate` (D-078, VALIDATION.md Report format) on seat 4's synthetic rig (tests/test_validation_plate.py)."""
import json

from test_validation_plate import DISTS, rig  # noqa: F401  (seat 4's fixture, reused rather than duplicated)

from ringsim.cli import main


def test_validate_plate_writes_report(rig, tmp_path, capsys):  # noqa: F811
    sim, ses = rig
    out = tmp_path / "reports" / "synthetic"
    assert main(["validate", "plate", "--sim", sim, "--real", ses, "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((out / "report.json").read_text())
    assert printed == saved and saved["peak_bin_pass"]["ref"] is True
    files = sorted(p.name for p in out.iterdir())
    assert files == ["differential_raw.png", "differential_ref.png", "mean_profiles_raw.png", "mean_profiles_ref.png",
                     "report.json", "report.md"]
    md = (out / "report.md").read_text()
    for d in DISTS:
        assert f"| {d} | ref |" in md and f"| {d} | raw |" in md
    assert "pooled over takes" in md and "PASS" in md
    # numbers in report.md come from the report itself (spot checks)
    assert f"{saved['noise_floor']:.5f}" in md
    assert f"{saved['linearity']['ref']['slope_bins_per_m']:.2f}" in md
    assert (out / "mean_profiles_ref.png").stat().st_size > 0
