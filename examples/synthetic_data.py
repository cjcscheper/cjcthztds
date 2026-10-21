"""Synthetic traces shared by the examples: no real data, but numbers a physicist can check.

- The reference is a single-cycle THz pulse, the derivative of a Gaussian
  (peaks 0.15 ps either side of its centre), as measured without a sample.
- The sample trace mimics the same pulse after a lossless silicon wafer
  (n = 3.42, 0.3 mm thick, normal incidence), as two copies of the pulse:
    - the direct pass, delayed by (n - 1) d / c = 2.42 ps and scaled by the two
      interface transmissions 4n / (n + 1)^2 = 0.70;
    - the first echo (two extra internal reflections), a further 2 n d / c = 6.84 ps
      later and scaled by r^2 = ((n - 1) / (n + 1))^2 = 0.30 more.
- Both get detector noise of 0.2 % of the pulse peak, from a fixed seed.

Run from any folder: the examples import this file from their own folder.
"""

import numpy as np

from cjcthztds import TimeTrace

SPEED_OF_LIGHT = 299_792_458.0  # m/s
N_SILICON, THICKNESS = 3.42, 0.3e-3  # refractive index, m

# Expected values, used by the examples to draw what the analysis should find.
TRANSMISSION = 4 * N_SILICON / (N_SILICON + 1) ** 2
ECHO_FACTOR = ((N_SILICON - 1) / (N_SILICON + 1)) ** 2
DELAY_PS = (N_SILICON - 1) * THICKNESS / SPEED_OF_LIGHT * 1e12
ECHO_DELAY_PS = 2 * N_SILICON * THICKNESS / SPEED_OF_LIGHT * 1e12

# 40 ps scan in 20 fs steps, like a typical TDS measurement.
TIME_PS = -10.0 + 0.02 * np.arange(2000)
NOISE = 2e-3


def single_cycle_pulse(time_ps: np.ndarray, center_ps: float = 0.0) -> np.ndarray:
    """Derivative of a Gaussian with peaks of +-1, 0.15 ps either side of ``center_ps``."""
    x = (time_ps - center_ps) / 0.15
    return -x * np.exp(0.5 * (1.0 - x**2))


def synthetic_data_reference_and_sample(seed: int = 0) -> tuple[TimeTrace, TimeTrace]:
    """The reference (no sample) and the sample trace (silicon wafer), as TimeTraces on TIME_PS."""
    rng = np.random.default_rng(seed)
    reference = single_cycle_pulse(TIME_PS) + NOISE * rng.standard_normal(TIME_PS.size)
    sample = TRANSMISSION * (
        single_cycle_pulse(TIME_PS, DELAY_PS) + ECHO_FACTOR * single_cycle_pulse(TIME_PS, DELAY_PS + ECHO_DELAY_PS)
    ) + NOISE * rng.standard_normal(TIME_PS.size)
    return TimeTrace(TIME_PS, reference), TimeTrace(TIME_PS, sample)
