from abc import ABC, abstractmethod

from rotorpy.sensors.imu import Imu
from rotorpy.vehicles.multirotor import Multirotor


class CrazyflieBase(ABC):
    @abstractmethod
    def build(self) -> Multirotor: ...


class IMUSensorBase(ABC):
    def __init__(self):
        self.sampling_rate = None

    @abstractmethod
    def build(self) -> Imu: ...
