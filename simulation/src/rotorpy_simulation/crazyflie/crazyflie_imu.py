from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rotorpy.sensors.imu import Imu

from ..base import IMUSensorBase


@dataclass
class CrazyflieIMU(IMUSensorBase):
    seed: int | None = None
    accel_bias: np.ndarray = field(init=False)
    gyro_bias: np.ndarray = field(init=False)
    accelerometer_params: Any = field(init=False)
    gyroscope_params: Any = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.accel_bias = rng.normal(0.0, 0.1962 / 3.0, size=3)  # fixed bias of +- 20 mg -> 0.1962 m/s^2
        self.gyro_bias = rng.normal(0.0, 0.01745 / 3.0, size=3)  # fixed bias of +- 1 dps -> 0.01745 rad/s
        self.accelerometer_params = {'initial_bias': self.accel_bias,  # m/s^2
                                     'noise_density': np.array([0.0016, 0.0016, 0.001864]),  # m/s^2 / sqrt(Hz)
                                     'random_walk': 3.0e-04 * np.ones(3, ),  # m/s^2 * sqrt(Hz) -> assumption
                                     }

        self.gyroscope_params = {'initial_bias': self.gyro_bias,  # rad/s
                                 'noise_density': 0.000223 * np.ones(3, ),  # rad/s / sqrt(Hz)
                                 'random_walk': 2.0e-05 * np.ones(3, ),  # rad/s * sqrt(Hz) -> assumption
                                 }

    R_BS = np.eye(3)
    p_BS = np.zeros(3, )
    sampling_rate = 1000
    gravity_vector = np.array([0, 0, -9.81])

    def build(self):
        return Imu(self.accelerometer_params,
                   self.gyroscope_params,
                   self.R_BS,
                   self.p_BS,
                   self.sampling_rate,
                   self.gravity_vector)
