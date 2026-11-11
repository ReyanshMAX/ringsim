"""Sound-speed constants shared by solver, dataset and proc (PROCESSING.md, D-037)."""
from __future__ import annotations

import numpy as np

C_SIM = 343.0                       # every k-Wave sim runs at this speed
TEMP_RANGE_C = (10.0, 35.0)         # supported read-time temperatures


def speed_of_sound(temp_c: float) -> float:
    return float(331.3 * np.sqrt(1 + temp_c / 273.15))


C_MIN = speed_of_sound(TEMP_RANGE_C[0])   # ~337.3 m/s
C_MAX = speed_of_sound(TEMP_RANGE_C[1])   # ~351.9 m/s, sets the sim time window (D-044)


def temperature_alpha(temp_c: float) -> float:
    """alpha = c(T) / C_SIM, the read-time time-scaling factor (D-037)."""
    assert TEMP_RANGE_C[0] <= temp_c <= TEMP_RANGE_C[1], temp_c
    return speed_of_sound(temp_c) / C_SIM
