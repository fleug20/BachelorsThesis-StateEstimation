from abc import ABC, abstractmethod

from core import GroundTruth
from evaluation.core.kalman_estimate import KalmanEstimate


class TimeAligner(ABC):
    """Aligns time of groundtruth and estimate to make them comparable."""

    @abstractmethod
    def align(self, estimate: KalmanEstimate, ground_truth: GroundTruth) -> tuple[KalmanEstimate, GroundTruth]:
        pass

    def describe(self) -> str:
        return type(self).__name__
