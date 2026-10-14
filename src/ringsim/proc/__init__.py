"""Shared signal processing (PROCESSING.md, D-008).

The only processing code in the project: sim IRs and real captures both pass through
here. Depends on numpy/scipy only; never imports ringsim.geometry, solver, dataset or
validation (tests/test_proc_isolation.py).
"""
from ringsim.proc.chirp import ChirpSpec, export_sweep_wav, export_tx_wav, synth_chirp, tx_frame
from ringsim.proc.ir import overlap_add_stream, rx_from_ir, temperature_scale
from ringsim.proc.physics import C_MIN, C_SIM, TEMP_RANGE_C, speed_of_sound, temperature_alpha
from ringsim.proc.profile import (align_direct_path, bandpass_sos, cfmcw_profiles, differential,
                                  frame_stream, magnitude)

__all__ = [
    "C_MIN", "C_SIM", "TEMP_RANGE_C", "ChirpSpec", "align_direct_path", "bandpass_sos",
    "cfmcw_profiles", "differential", "export_sweep_wav", "export_tx_wav", "frame_stream",
    "magnitude", "overlap_add_stream", "rx_from_ir", "speed_of_sound", "synth_chirp",
    "temperature_alpha", "temperature_scale", "tx_frame",
]
