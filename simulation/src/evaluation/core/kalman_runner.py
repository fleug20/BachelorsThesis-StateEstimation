from abc import ABC, abstractmethod

from core import SimulationResult
from evaluation.core.kalman_estimate import KalmanEstimate


class KalmanRunner(ABC):
    """Runs a Kalman filter against a SimulationResult and returns its KalmanEstimate

    NOTE: the runner is responsible for providing data in the correct unit and coordinate frame!
    """

    @abstractmethod
    def run(self, simulation_result: SimulationResult) -> KalmanEstimate:
        pass

    def describe(self) -> str:
        return type(self).__name__
