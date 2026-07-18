import numpy as np

from core import GroundTruth
from evaluation.core.kalman_estimate import KalmanEstimate
from evaluation.core.metric import Metric, MetricResult


class PositionRMSE(Metric):
    """Root-mean-square error on 3D position in meters."""

    def compute(self, gt: GroundTruth, est: KalmanEstimate) -> MetricResult:
        diff = est.position - gt.position  # (N, 3)
        per_axis = np.sqrt((diff ** 2).mean(axis=0))
        total = float(np.sqrt((diff ** 2).sum(axis=1).mean()))
        return MetricResult(
            name="position_rmse",
            value=total,
            unit="m",
            per_axis={
                "x": float(per_axis[0]),
                "y": float(per_axis[1]),
                "z": float(per_axis[2]),
            },
        )
