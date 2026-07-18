import numpy as np

from core import SimulationResult
from evaluation.core.kalman_estimate import KalmanEstimate
from evaluation.core.kalman_runner import KalmanRunner


class PassthroughRunner(KalmanRunner):
    """ Runner just for testing... takes a SimulationResult and returns it as KalmanEstimate with optional jitter and downsampling."""

    def __init__(self, jitter_std: float = 0.0, downsample_factor: int = 1, seed: int | None = 42, ):
        if jitter_std < 0:
            raise ValueError(f"PassthroughRunner: jitter_std must be >= 0, got {jitter_std}")
        self._jitter_std = float(jitter_std)
        self._downsample_factor = downsample_factor
        self._seed = seed

    def run(self, simulation_result: SimulationResult) -> KalmanEstimate:
        gt = simulation_result.ground_truth
        rng = np.random.default_rng(self._seed)

        # downsample first so jitter only gets applied to the kept samples
        step = self._downsample_factor
        time = gt.time[::step].copy()
        position = gt.position[::step].copy()
        geo_coordinates = gt.geo_coordinates[::step].copy()
        velocity = gt.velocity[::step].copy()
        acceleration = gt.acceleration[::step].copy()
        attitude = gt.attitude[::step].copy()
        angular_velocity = gt.angular_velocity[::step].copy()
        pressure = gt.pressure[::step].copy()
        output_rate = gt.simulation_sampling_rate / step

        if self._jitter_std > 0:
            position += rng.normal(0.0, self._jitter_std, size=position.shape)
            velocity += rng.normal(0.0, self._jitter_std, size=velocity.shape)
            acceleration += rng.normal(0.0, self._jitter_std, size=acceleration.shape)
            angular_velocity += rng.normal(0.0, self._jitter_std, size=angular_velocity.shape)

        return KalmanEstimate(
            time=time,
            position=position,
            geo_coordinates=geo_coordinates,
            velocity=velocity,
            acceleration=acceleration,
            attitude=attitude,
            angular_velocity=angular_velocity,
            pressure=pressure,
            estimation_output_data_rate=output_rate,
            metadata={
                "runner": "PassthroughRunner",
                "runner_config": self.describe(),
                "seed": self._seed,
            },
        )

    def describe(self) -> str:
        return (
            f"PassthroughRunner(jitter_std={self._jitter_std}, "
            f"downsample_factor={self._downsample_factor}, seed={self._seed})"
        )
