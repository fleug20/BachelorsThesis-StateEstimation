from abc import ABC, abstractmethod
from dataclasses import dataclass

from core import GroundTruth
from evaluation.core.kalman_estimate import KalmanEstimate


@dataclass
class MetricResult:
    """Result produced by a Metric.

    value: the scalar result.
    per_axis: optionally breaks the value down.
    """

    name: str
    value: float
    unit: str
    per_axis: dict[str, float] | None = None


class Metric(ABC):
    """Single metric comparing a KalmanEstimate against a GroundTruth.

    Implementations are allowed to assume gt and est are already time-aligned.
    """

    @abstractmethod
    def compute(self, gt: GroundTruth, est: KalmanEstimate) -> MetricResult:
        pass

    def describe(self) -> str:
        return type(self).__name__
