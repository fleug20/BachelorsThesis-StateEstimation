import numpy as np
from scipy.spatial.transform import Rotation as R

from core import GroundTruth
from evaluation.core import Metric, KalmanEstimate, MetricResult


class NEES(Metric):
    """Normalized Estimation Error Squared, averaged over time.

    A consistent filter has mean(epsilon) ≈ n (state dimension).
    - if nees > n: filter is overconfident
    - if nees < n: filter is underconfident
    """

    def compute(self, gt: GroundTruth, est: KalmanEstimate) -> MetricResult:
        # Attitude error as rotation vector (small-angle representation)
        r_err = (
                R.from_quat(est.attitude, scalar_first=True)
                * R.from_quat(gt.attitude, scalar_first=True).inv()
        )
        att_err = r_err.as_rotvec()

        # Full error vector: [pos, vel, att]
        err = np.concatenate(
            [est.position - gt.position, est.velocity - gt.velocity, att_err],
            axis=1,
        )  # (N, 9)

        P = est.covariance  # (N, 9, 9)
        sol = np.linalg.solve(P, err[..., None]).squeeze(-1)
        epsilon = np.einsum("ni,ni->n", err, sol)  # (N,)

        return MetricResult(
            name="NEES",
            value=float(epsilon.mean()),
            unit="",
        )
