import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from evaluation.core import EvaluationResult

_IDX_POS = slice(0, 3)
_IDX_VEL = slice(3, 6)
_IDX_ATT = slice(6, 9)


def _nees_per_timestep(err: np.ndarray, P: np.ndarray) -> np.ndarray:
    sol = np.linalg.solve(P, err[..., None]).squeeze(-1)
    return np.einsum("ni,ni->n", err, sol)


def plot_nees_groups_over_time(result: EvaluationResult, ax: plt.Axes | None = None) -> plt.Axes:
    """Position, velocity and attitude NEES over time"""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    t = result.estimate.time
    gt = result.ground_truth
    est = result.estimate
    P = est.covariance  # (N, 9, 9)

    pos_err = est.position - gt.position
    nees_pos = _nees_per_timestep(pos_err, P[:, _IDX_POS, _IDX_POS])

    vel_err = est.velocity - gt.velocity
    nees_vel = _nees_per_timestep(vel_err, P[:, _IDX_VEL, _IDX_VEL])

    r_err = (
            Rotation.from_quat(est.attitude, scalar_first=True)
            * Rotation.from_quat(gt.attitude, scalar_first=True).inv()
    )
    att_err = r_err.as_rotvec()  # (N, 3)
    nees_att = _nees_per_timestep(att_err, P[:, _IDX_ATT, _IDX_ATT])

    ax.plot(t, nees_pos, label="Position", linewidth=1)
    ax.plot(t, nees_vel, label="Velocity", linewidth=1)
    ax.plot(t, nees_att, label="Attitude", linewidth=1)
    ax.axhline(3, color="gray", linestyle="--", linewidth=0.8, label="Expected (dof = 3)")
    # alpha = 0.05
    # r1 = chi2.ppf(alpha / 2, 3)
    # r2 = chi2.ppf(1 - alpha / 2, 3)
    # ax.axhline(r1, color="red", linestyle="--", linewidth=0.8, label="Confidence interval")
    # ax.axhline(r2, color="red", linestyle="--", linewidth=0.8,)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("NEES")
    ax.set_title("Grouped NEES over time")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax
