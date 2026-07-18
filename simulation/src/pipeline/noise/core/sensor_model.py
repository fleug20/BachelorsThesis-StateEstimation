import numpy as np

from .noise_profile import NoiseProfile


class SensorModel:
    """Applies realistic sensor noise to a clean 1D signal.
    Partially based on Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems by Groves, Paul D

    Order of application:
        1. Scale factor error
        2. Fixed bias
        3. Random walk
        4. White noise
        5. Clipping range
        6. Resolution (if set)
    """

    def __init__(self, profile: NoiseProfile, seed: int | None = None):
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        self.bias_fixed = self.rng.normal(0.0, self.profile.bias_fixed / 3.0)

    def apply(self, signal: np.ndarray, dt: float) -> np.ndarray:
        """Apply noise in-place and return the array."""
        n = len(signal)

        # 1. SCALE FACTOR ERROR
        signal *= (1.0 + self.profile.scale_factor_error)

        # 2. FIXED BIAS -> constant per run (fixed offset that is always present). Is sampled from a Gaussian with 3 sigma
        signal += self.bias_fixed

        # 3. RANDOM WALK -> this is the in-run bias variation. Watch out to pass the correct parameter here as some ambiguity exists between random walk and bias instability...
        walk_steps = self.rng.normal(0, self.profile.random_walk * np.sqrt(dt), n)
        signal += np.cumsum(walk_steps)

        # 4. WHITE NOISE (per sample jitter)
        sample_rate = 1.0 / dt
        sigma = self.profile.noise_density * np.sqrt(sample_rate / 2)  # = noise_density * sqrt(ODR / 2)
        signal += self.rng.normal(0, sigma, n)

        # 5. CLIPPING RANGE
        signal[:] = np.clip(signal, self.profile.range_min, self.profile.range_max)

        # 6. RESOLUTION (if set)
        if self.profile.resolution is not None:
            signal[:] = np.round(signal / self.profile.resolution) * self.profile.resolution

        return signal
