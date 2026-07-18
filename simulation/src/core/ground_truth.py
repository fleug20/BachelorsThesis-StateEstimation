import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GroundTruth:
    """True state over time (Ground truth).

    All arrays share the same first dimension *N* (number of time steps).
    The coordinate frame is the simulation's inertial (world) frame.
    """

    def __post_init__(self):
        self.validate()

    time: np.ndarray  # s
    position: np.ndarray  # m, shape (N, 3), ENU
    geo_coordinates: np.ndarray  # [deg, deg, m above sea level], shape (N, 3)
    velocity: np.ndarray  # m/s, shape (N, 3), ENU
    acceleration: np.ndarray  # m/s², shape (N, 3), ENU
    attitude: np.ndarray  # quaternion [w, x, y, z], scalar-first, BODY -> ENU
    angular_velocity: np.ndarray  # rad/s, shape (N, 3), body frame
    pressure: np.ndarray  # Pa, shape (N,),
    simulation_sampling_rate: float  # Hz

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "time": self.time,
            "x": self.position[:, 0],
            "y": self.position[:, 1],
            "z": self.position[:, 2],
            "lat": self.geo_coordinates[:, 0],
            "lon": self.geo_coordinates[:, 1],
            "alt": self.geo_coordinates[:, 2],
            "vx": self.velocity[:, 0],
            "vy": self.velocity[:, 1],
            "vz": self.velocity[:, 2],
            "ax": self.acceleration[:, 0],
            "ay": self.acceleration[:, 1],
            "az": self.acceleration[:, 2],
            "qw": self.attitude[:, 0],
            "qx": self.attitude[:, 1],
            "qy": self.attitude[:, 2],
            "qz": self.attitude[:, 3],
            "wx": self.angular_velocity[:, 0],
            "wy": self.angular_velocity[:, 1],
            "wz": self.angular_velocity[:, 2],
            "pressure": self.pressure,
        })
        return df

    def validate(self) -> None:
        """Check ground truth data for obvious errors."""
        n = len(self.time)

        # Shape checks
        if self.time.ndim != 1:
            raise ValueError(f"time should be 1D, got {self.time.ndim}D")
        if self.position.shape != (n, 3):
            raise ValueError(f"position shape {self.position.shape}, expected ({n}, 3)")
        if self.geo_coordinates.shape != (n, 3):
            raise ValueError(f"geo_coordinates shape {self.geo_coordinates.shape}, expected ({n}, 3)")
        if self.velocity.shape != (n, 3):
            raise ValueError(f"velocity shape {self.velocity.shape}, expected ({n}, 3)")
        if self.acceleration.shape != (n, 3):
            raise ValueError(f"acceleration shape {self.acceleration.shape}, expected ({n}, 3)")
        if self.attitude.shape != (n, 4):
            raise ValueError(f"attitude shape {self.attitude.shape}, expected ({n}, 4)")
        if self.angular_velocity.shape != (n, 3):
            raise ValueError(f"angular_velocity shape {self.angular_velocity.shape}, expected ({n}, 3)")
        if self.pressure.shape != (n,):
            raise ValueError(f"pressure shape {self.pressure.shape}, expected ({n},)")

        # Time checks
        if not np.all(np.diff(self.time) > 0):
            raise ValueError("time must be strictly monotonically increasing")

        # NaN / Inf checks ---
        for name, arr in [
            ("position", self.position),
            # ("geo_coordinates", self.geo_coordinates), # is null for crazyflie
            ("velocity", self.velocity),
            ("acceleration", self.acceleration),
            ("attitude", self.attitude),
            ("angular_velocity", self.angular_velocity),
            # ("pressure", self.pressure), # is null for crazyflie
        ]:
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or Inf")

        # --- Quaternion checks ---
        norms = np.linalg.norm(self.attitude, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                f"Quaternions not normalized, norm range "
                f"[{norms.min():.4f}, {norms.max():.4f}]"
            )

        # scalar-first convention
        w_positive_ratio = np.mean(self.attitude[:, 0] >= 0)
        if w_positive_ratio < 0.5:
            warnings.warn(
                f"Quaternion w is negative {(1 - w_positive_ratio) * 100:.1f}% of the time. "
                f"This is valid but may indicate a sign convention mismatch."
            )

        # magnitude checks
        max_speed = np.max(np.linalg.norm(self.velocity, axis=1))
        if max_speed >= 500.0:
            warnings.warn(f"Max speed {max_speed:.1f} m/s exceeds 500 m/s, likely an error")

        max_accel = np.max(np.linalg.norm(self.acceleration, axis=1))
        if max_accel >= 1000.0:
            raise ValueError(f"Max acceleration {max_accel:.1f} m/s² exceeds 1000 m/s², likely an error")

        max_angular_vel = np.max(np.abs(self.angular_velocity))
        if max_angular_vel >= 100.0:
            raise ValueError(f"Max angular velocity {max_angular_vel:.1f} rad/s exceeds 100 rad/s, likely an error")

        print("GroundTruth: validation passed!")
