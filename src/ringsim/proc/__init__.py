"""Shared signal processing (PROCESSING.md, D-008).

The only processing code in the project: simulated IRs and real captures both pass through
here. Depends on numpy and scipy only; never imports ringsim.geometry, solver, dataset or
validation (tests/test_proc_isolation.py, D-045).
"""
from ringsim.proc.chirp import ChirpSpec, export_sweep_wav, export_tx_wav, synth_chirp, tx_frame
from ringsim.proc.ir import rx_from_ir, rx_stream, snapshot_weights, temperature_scale
from ringsim.proc.physics import (C_MAX, C_MIN, C_SIM, TEMP_RANGE_C, speed_of_sound,
                                  temperature_alpha)
from ringsim.proc.profile import (align_direct_path, cfmcw_profiles, cfmcw_profiles_ref,
                                  differential, frame_stream, free_air_reference, magnitude)

__all__ = [
    "C_MAX", "C_MIN", "C_SIM", "TEMP_RANGE_C", "ChirpSpec", "align_direct_path",
    "cfmcw_profiles", "cfmcw_profiles_ref", "differential", "export_sweep_wav", "export_tx_wav",
    "frame_stream", "free_air_reference", "magnitude", "rx_from_ir", "rx_stream",
    "snapshot_weights", "speed_of_sound", "synth_chirp", "temperature_alpha",
    "temperature_scale", "tx_frame",
]
