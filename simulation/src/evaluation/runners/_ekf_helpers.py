"""Shared utilities for EKF runner implementations."""

import numpy as np
import pymap3d as pm
from ekf import Ekf, GeodeticOrigin, STATE_DIM
from scipy.spatial.transform import Rotation

from core.sensor_stream import SensorStream

# 180° rotation about (1, 1, 0)/√2 — maps ENU frame to NED frame.
_R_ENU_TO_NED = Rotation.from_rotvec(np.pi * np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))


def _enu_to_ned_vec(v: np.ndarray) -> np.ndarray:
    """Convert a single [E, N, U] vector to [N, E, D]."""
    return np.array([v[1], v[0], -v[2]])


def _ned_to_enu_batch(arr: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) NED array to ENU: swap first two columns, negate third."""
    return arr[:, [1, 0, 2]] * np.array([1.0, 1.0, -1.0])


def _ned_to_geodetic(pos_ned: np.ndarray, origin: GeodeticOrigin, origin_alt_above_ground: float) -> np.ndarray:
    """Convert (N, 3) NED positions to [lat_deg, lon_deg, alt_above_ground_m]."""
    lat, lon, alt_abs = pm.ned2geodetic(
        pos_ned[:, 0], pos_ned[:, 1], pos_ned[:, 2],
        origin.lat_deg, origin.lon_deg, origin.alt_m,
        deg=True,
    )
    alt_above_ground = alt_abs - (origin.alt_m - origin_alt_above_ground)
    return np.column_stack([lat, lon, alt_above_ground])


def _find_sensor(sensors: dict[str, SensorStream], keyword: str, required: bool = True) -> str | None:
    """Return the unique sensor key containing *keyword* (case-insensitive)"""
    matches = [k for k in sensors if keyword.lower() in k.lower()]
    if not matches:
        if required:
            raise KeyError(f"No sensor containing '{keyword}' found. Available: {list(sensors.keys())}")
        return None
    if len(matches) > 1:
        raise KeyError(
            f"Multiple sensors match '{keyword}': {matches}. Specify the key explicitly via the constructor.")
    return matches[0]


def _init_ekf_from_gt_asteria(gt) -> Ekf:
    """Initialise an Ekf instance from the first ground-truth sample."""
    ekf = Ekf()

    ekf.state = np.concatenate([
        _enu_to_ned_vec(gt.position[0]),
        _enu_to_ned_vec(gt.velocity[0]),
        np.zeros(3),
    ])
    ekf.attitude = (_R_ENU_TO_NED * Rotation.from_quat(gt.attitude[0], scalar_first=True)).as_quat(scalar_first=True)

    cov0 = np.zeros((STATE_DIM, STATE_DIM))
    cov0[0, 0] = cov0[1, 1] = cov0[2, 2] = 0.01
    cov0[3, 3] = cov0[4, 4] = cov0[5, 5] = 0.001
    cov0[6, 6] = cov0[7, 7] = cov0[8, 8] = 0.001
    ekf.covariance = cov0

    return ekf


def _init_ekf_from_gt_crazyflie(gt) -> Ekf:
    """Initialise an Ekf instance from the first ground-truth sample."""
    ekf = Ekf()
    ekf.state = np.concatenate([
        _enu_to_ned_vec(gt.position[0]),
        _enu_to_ned_vec(gt.velocity[0]),
        np.zeros(3),
    ])

    _R_FLU_TO_FRD = Rotation.from_matrix(np.diag([1.0, -1.0, -1.0]))
    gt_att = Rotation.from_quat(gt.attitude[0], scalar_first=True)

    # Init as FRD -> NED
    ekf_start_att = _R_ENU_TO_NED * gt_att * _R_FLU_TO_FRD
    ekf.attitude = ekf_start_att.as_quat(scalar_first=True)

    cov0 = np.zeros((STATE_DIM, STATE_DIM))
    cov0[0, 0] = cov0[1, 1] = cov0[2, 2] = 0.01
    cov0[3, 3] = cov0[4, 4] = cov0[5, 5] = 0.001
    cov0[6, 6] = cov0[7, 7] = cov0[8, 8] = 0.001
    ekf.covariance = cov0

    return ekf
