import numpy as np
import pytest

from ringsim.proc import ChirpSpec

TEST_CHIRP = ChirpSpec(f0_hz=18000, f1_hz=22000, duration_s=0.0125,
                       frame_period_s=0.0125, window="none")   # test_pair / phone (D-022)
FS = 48000
BAND = (18000.0, 22000.0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
