"""Domain randomization (DATASET.md, D-013). Phase 1: load-time noise only."""
from __future__ import annotations

import numpy as np


def add_noise(stream: np.ndarray, snr_db: float, ref_power: float,
              rng: np.random.Generator) -> np.ndarray:
    """AWGN at snr_db relative to ref_power (direct-path power in band)."""
    sigma = np.sqrt(ref_power / 10 ** (snr_db / 10))
    return (stream + sigma * rng.standard_normal(len(stream))).astype(np.float32)
