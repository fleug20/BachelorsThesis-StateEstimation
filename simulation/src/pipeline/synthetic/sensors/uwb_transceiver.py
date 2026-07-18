import math

import numpy as np

from core import GroundTruth, SensorStream, SensorOrigin
from pipeline.synthetic import SyntheticSensor


class SyntheticUWBTransceiver(SyntheticSensor):
    def __init__(self, name: str = "LPSdeck", sampling_rate: float = 50,
                 anchor_position: np.ndarray = np.array([0, 0, 0], ), anchor_id: int = -1, ):
        self.name = name
        self.sampling_rate = sampling_rate
        self.anchor_position = anchor_position
        self.anchor_id = anchor_id

    def generate(self, ground_truth: GroundTruth) -> SensorStream:
        self._check_sampling_rate(ground_truth.simulation_sampling_rate)
        step: int = int(ground_truth.simulation_sampling_rate / self.sampling_rate)

        time = ground_truth.time[::step].copy()
        n = len(time)
        channels = ["distance", "anchor_id", "anchor_x", "anchor_y", "anchor_z"]
        positions = ground_truth.position[::step].copy()  # (N, 3)
        distances = self._get_distance(positions, self.anchor_position)  # (N,)
        anchor_ids = np.full(n, self.anchor_id, dtype=float)  # (N,)
        anchor_positions = np.tile(self.anchor_position, (n, 1))  # (N, 3)
        unit = "m"
        origin = SensorOrigin.SYNTHETIC
        processing_history = [
            f"Sensor data by SyntheticSensorGenerator. Anchor ID: {self.anchor_id}, "
            f"Anchor position: {self.anchor_position}, Sensor origin: {origin}, Sampling rate: {str(self.sampling_rate)} Hz, Data units: {unit}"
        ]

        return SensorStream(name=self.name,
                            time=time,
                            data=np.column_stack(
                                [distances, anchor_ids, anchor_positions]),
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

    def _get_distance(self, position: np.ndarray, anchor_position: np.ndarray) -> np.ndarray:
        return np.linalg.norm(anchor_position - position, axis=1)
