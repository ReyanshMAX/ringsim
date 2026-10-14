"""Solver constants (SOLVER.md)."""
import numpy as np

from ringsim.proc.physics import C_MIN, C_SIM, speed_of_sound  # noqa: F401  (D-037)

PULSE_FC_HZ = 22_000.0
PULSE_HALF_BW = 6_000.0          # -6 dB at 16 and 28 kHz
SIGMA_T = np.sqrt(2 * np.log(2)) / (2 * np.pi * PULSE_HALF_BW)   # ~31.2e-6 s
PULSE_DELAY = 4 * SIGMA_T        # ~125e-6 s
CFL = 0.3
PML_SIZE = 10                    # cells, outside the domain (pml_inside=False)
T_MARGIN_S = 0.3e-3
DOMAIN_MARGIN_M = 0.010          # air padding around solids and ports
PPW_MIN = 4                      # points per wavelength at 28 kHz, asserted
F_MAX_HZ = 28_000.0
