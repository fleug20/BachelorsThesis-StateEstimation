from dataclasses import dataclass

import numpy as np
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor

from ..base import CrazyflieBase


@dataclass
class Crazyflie(CrazyflieBase):
    initial_state = {'x': np.array([0, 0, 0]),
                     'v': np.zeros(3, ),
                     'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
                     'w': np.zeros(3, ),
                     'wind': np.array([0, 0, 0]),  # Since wind is handled elsewhere, this value is overwritten
                     'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

    def build(self) -> Multirotor:
        return Multirotor(quad_params, self.initial_state)
