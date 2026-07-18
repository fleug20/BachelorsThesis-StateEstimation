import numpy as np

from core import GroundTruth
from evaluation.core import Metric, KalmanEstimate, MetricResult
from scipy.spatial.transform import Rotation as R


class AttitudeRMSE(Metric):
    """Root-mean-square error on 3D rotation in degrees."""

    def compute(self, gt: GroundTruth, est: KalmanEstimate) -> MetricResult:
        rotation_gt = R.from_quat(gt.attitude, scalar_first=True)
        rotation_est = R.from_quat(est.attitude, scalar_first=True)

        # This represents the angular difference between the two quaternions
        # https://stackoverflow.com/questions/57063595/how-to-obtain-the-angle-between-two-quaternions
        r_err = rotation_gt * rotation_est.inv()

        # angle (in radiant) of the rotation
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.magnitude.html#scipy.spatial.transform.Rotation.magnitude
        angles_rad = r_err.magnitude()  # (N, 4)

        total_rad = float(np.sqrt((angles_rad ** 2).mean()))
        total_deg = float(np.degrees(total_rad))

        # per-axis breakdown via the rotation vector
        rotation_vec = r_err.as_rotvec()  # (N, 3), radians
        per_axis_rad = np.sqrt((rotation_vec ** 2).mean(axis=0))
        per_axis_deg = np.degrees(per_axis_rad)

        return MetricResult(
            name="attitude_rmse",
            value=total_deg,
            unit="deg",
            per_axis={
                "x": float(per_axis_deg[0]),
                "y": float(per_axis_deg[1]),
                "z": float(per_axis_deg[2]),
            },
        )
