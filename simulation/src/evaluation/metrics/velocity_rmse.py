import numpy as np

from core import GroundTruth
from evaluation.core import Metric, KalmanEstimate, MetricResult


class VelocityRMSE(Metric):
    """Root-mean-square error on 3D velocity in m/s."""

    def compute(self, gt: GroundTruth, est: KalmanEstimate) -> MetricResult:
        diff = est.velocity - gt.velocity  # (N, 3)
        per_axis = np.sqrt((diff ** 2).mean(axis=0))
        total = float(np.sqrt((diff ** 2).sum(axis=1).mean()))
        return MetricResult(
            name="velocity_rmse",
            value=total,
            unit="m/s",
            per_axis={
                "x": float(per_axis[0]),
                "y": float(per_axis[1]),
                "z": float(per_axis[2]),
            },
        )
