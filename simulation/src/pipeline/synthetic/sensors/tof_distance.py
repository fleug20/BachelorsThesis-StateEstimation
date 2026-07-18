import math

import numpy as np

from core import GroundTruth, SensorStream, SensorOrigin
from pipeline.synthetic import SyntheticSensor
from scipy.spatial.transform import Rotation as R


class SyntheticTimeOfFlightDistance(SyntheticSensor):
    def __init__(self, name: str = "time_of_flight_distance", sampling_rate: float = 40.0):
        self.name = name
        self.sampling_rate = sampling_rate

    def generate(self, ground_truth: GroundTruth) -> SensorStream:
        self._check_sampling_rate(ground_truth.simulation_sampling_rate)
        step: int = int(ground_truth.simulation_sampling_rate / self.sampling_rate)

        time = ground_truth.time[::step].copy()
        altitudes = ground_truth.position[::step, 2].copy()
        quaternions = ground_truth.attitude[::step].copy()

        rotations = R.from_quat(quaternions, scalar_first=True)
        r_zz = rotations.as_matrix()[:, 2, 2]  # Z-Z component of the rotation
        r_zz_safe = np.clip(r_zz, -1.0, 1.0)  # fix for floating point errors
        drone_tilt_angle = np.arccos(r_zz_safe)  # actual angle of the drone
        tilt_raw = np.abs(drone_tilt_angle) - math.radians(7.5)  # tilt compensated for cone.
        tilt = np.maximum(0.0, tilt_raw)  # altitudes grow if drone is tilted more than 7.5 degrees. 0 if drone is tilted less than 7.5 degrees.
        distances = altitudes / np.cos(tilt)

        channels = ["altitude"]
        unit = "m"
        origin = SensorOrigin.SYNTHETIC
        processing_history = [
            f"Sensor data by SyntheticSensorGenerator. Sensor origin: {origin}, Sampling rate: {str(self.sampling_rate)} Hz, Data units: {unit}"
        ]

        return SensorStream(name=self.name,
                            time=time,
                            data=distances,
                            channels=channels,
                            unit=unit,
                            sampling_rate=self.sampling_rate,
                            origin=origin,
                            processing_history=processing_history)

    def _check_sampling_rate(self, simulation_sampling_rate: float):
        ratio = simulation_sampling_rate / self.sampling_rate
        if not math.isclose(ratio, round(ratio)):
            raise ValueError(
                f"Interpolation of Ground Truth not implemented yet... Ground truth sampling rate must be a multiple of the sensor sampling rate ({self.sampling_rate})"
            )
