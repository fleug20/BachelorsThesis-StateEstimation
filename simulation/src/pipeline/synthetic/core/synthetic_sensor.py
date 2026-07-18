from abc import ABC, abstractmethod

from core import GroundTruth
from core import SensorStream


class SyntheticSensor(ABC):

    @abstractmethod
    def generate(self, ground_truth: GroundTruth) -> SensorStream:
        pass
