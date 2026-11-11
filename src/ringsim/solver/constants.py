"""Solver constants (docs/SOLVER.md, Constants)."""
import numpy as np

from ringsim.proc.physics import C_MAX, C_SIM, speed_of_sound  # noqa: F401  (D-037: re-exported for solver code)

PULSE_FC_HZ = 22_000.0
PULSE_HALF_BW = 6_000.0          # -6 dB at 16 and 28 kHz
SIGMA_T = np.sqrt(2 * np.log(2)) / (2 * np.pi * PULSE_HALF_BW)   # ≈ 31.2e-6 s
PULSE_DELAY = 4 * SIGMA_T        # ≈ 125e-6 s
PULSE_TAIL = 4 * SIGMA_T         # an echo's pulse lasts until 4σ after its centre
CFL = 0.3
PML_SIZE = 10                    # cells, outside the domain (pml_inside=False)
T_MARGIN_S = 0.15e-3             # D-044
DOMAIN_MARGIN_M = 0.010          # air padding around solids and ports
PPW_MIN = 4                      # points per wavelength at 28 kHz, asserted
PPW_F_HZ = 28_000.0              # top of the pulse band, where PPW is checked

# Every sim runs at C_SIM = 343.0 m/s. Temperature is applied at read time by exact time-scaling of the
# impulse response (PROCESSING.md temperature_scale, D-037); t_end is sized for the hottest air, C_MAX (D-044).
