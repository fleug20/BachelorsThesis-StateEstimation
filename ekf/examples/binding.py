"""Test for the Python bindings.

Drives the filter through:
  - a short ConstantVelocity predict
  - a short IMU predict (stationary, specific-force convention)
  - a GNSS correction at the origin

and prints the resulting state, position, velocity, and attitude.
"""

import numpy as np

from ekf import (
    AccelConvention,
    ConstantVelocity,
    Ekf,
    GeodeticOrigin,
    Gnss,
    IDX_N,
    IDX_VN,
    Imu,
    ImuNoise,
    STATE_DIM,
)


def main() -> None:
    assert STATE_DIM == 9

    ekf = Ekf()
    print("initial state:", ekf.state)
    print("initial position:", ekf.position)
    print("initial attitude (w, x, y, z):", ekf.attitude)

    # Seed a 5 m/s northward velocity, then run a ConstantVelocity predict for 1 s.
    state = ekf.state.copy()
    state[IDX_VN] = 5.0
    ekf.state = state

    cv = ConstantVelocity(accel_stddev=0.5)
    for _ in range(100):
        ekf.predict(cv, 0.01)
    print(f"after 1 s CV predict: north = {ekf.position[0]:.3f} m (expected ~5.0)")

    # Stationary IMU: specific force = (0, 0, -g), gyro = 0.
    imu = Imu(
        accel=np.array([0.0, 0.0, -9.81]),
        gyro=np.zeros(3),
        noise=ImuNoise(accel_stddev=0.1, gyro_stddev=0.01),
        accel_convention=AccelConvention.SpecificForce,
    )
    for _ in range(100):
        ekf.predict(imu, 0.01)
    print(f"after stationary IMU: velocity = {ekf.velocity}")

    # GNSS fix at the origin should pull the filter back toward zero.
    origin = GeodeticOrigin(lat_deg=48.2082, lon_deg=16.3738, alt_m=170.0)
    gnss = Gnss.from_geodetic(
        origin.lat_deg,
        origin.lon_deg,
        origin.alt_m,
        origin,
        stddev=np.array([1.0, 1.0, 2.0]),
    )
    for _ in range(50):
        ekf.correct(gnss)
    print(f"after GNSS correction: position = {ekf.position}")

    print("covariance diag:", np.diag(ekf.covariance))
    print(f"position variance at N: {ekf.covariance[IDX_N, IDX_N]:.4f}")


if __name__ == "__main__":
    main()
