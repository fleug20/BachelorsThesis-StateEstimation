//! IMU-driven motion model
//!
//! Per step the user supplies specific-force and angular-rate readings in the
//! body frame. The nominal attitude is propagated directly into the filter's
//! quaternion; the attitude-error state stays at zero through predict (it only
//! becomes non-zero when an attitude-observing measurement fires a correction).

use nalgebra::{RealField, UnitQuaternion, Vector3};

use crate::cast;
use crate::predict::Predictor;
use crate::{CovMatrix, Ekf, IDX_AX, IDX_D, IDX_E, IDX_N, IDX_VD, IDX_VE, IDX_VN};

#[derive(Debug, Clone, Copy)]
pub struct ImuNoise<T: RealField + Copy = f64> {
    /// Accelerometer noise 1-sigma [m/s^2].
    accel_stddev: T,
    /// Gyroscope noise 1-sigma [rad/s].
    gyro_stddev: T,
}

impl<T: RealField + Copy> Default for ImuNoise<T> {
    fn default() -> Self {
        Self {
            accel_stddev: cast(0.0),
            gyro_stddev: cast(0.0),
        }
    }
}

impl<T: RealField + Copy> ImuNoise<T> {
    pub fn new(accel_stddev: T, gyro_stddev: T) -> Self {
        Self {
            accel_stddev,
            gyro_stddev,
        }
    }

    pub fn accel_stddev(&self) -> T {
        self.accel_stddev
    }

    pub fn gyro_stddev(&self) -> T {
        self.gyro_stddev
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum AccelConvention {
    /// Specific force in body frame: stationary level reads `(0, 0, −g)`
    #[default]
    SpecificForce,
    /// Coordinate acceleration in body frame: stationary level reads `(0, 0, 0)`
    Coordinate,
}

/// One IMU sample: body-frame accelerometer and gyroscope readings, plus noise params.
#[derive(Debug)]
pub struct Imu<T: RealField + Copy = f64> {
    /// Accelerometer reading (body frame) [m/s^2].
    accel: Vector3<T>,
    /// Gyroscope reading (body frame) [rad/s].
    gyro: Vector3<T>,
    noise: ImuNoise<T>,
    accel_convention: AccelConvention,
}

impl<T: RealField + Copy> Default for Imu<T> {
    fn default() -> Self {
        Self {
            accel: Vector3::zeros(),
            gyro: Vector3::zeros(),
            noise: ImuNoise::default(),
            accel_convention: AccelConvention::default(),
        }
    }
}

impl<T: RealField + Copy> Imu<T> {
    pub fn init(&mut self) {
        self.accel = Vector3::zeros();
        self.gyro = Vector3::zeros();
        self.noise = ImuNoise::default();
        self.accel_convention = AccelConvention::SpecificForce;
    }

    pub fn set_accel(mut self, accel: Vector3<T>) -> Self {
        self.accel = accel;
        self
    }

    pub fn set_accel_convention(mut self, accel_convention: AccelConvention) -> Self {
        self.accel_convention = accel_convention;
        self
    }

    pub fn set_gyro(mut self, gyro: Vector3<T>) -> Self {
        self.gyro = gyro;
        self
    }

    pub fn set_noise(mut self, noise: ImuNoise<T>) -> Self {
        self.noise = noise;
        self
    }

    pub fn update_measurement(&mut self, accel: Vector3<T>, gyro: Vector3<T>, noise: ImuNoise<T>) {
        self.accel = accel;
        self.gyro = gyro;
        self.noise = noise;
    }
}

impl<T: RealField + Copy> Predictor<T> for Imu<T> {
    fn apply(&mut self, ekf: &mut Ekf<T>, dt: T) {
        let rotation_matrix = ekf.rotation_matrix();

        let zero: T = cast(0.0);
        let half: T = cast(0.5);
        let two: T = cast(2.0);
        let four: T = cast(4.0);
        let gravity_z: T = cast(9.81);

        let dq_half = UnitQuaternion::from_scaled_axis(self.gyro * (half * dt));
        let rotation_matrix_mid = (ekf.attitude * dq_half).to_rotation_matrix();

        let kinematic_accelaration = match self.accel_convention {
            AccelConvention::SpecificForce => {
                let gravity_ned = Vector3::new(zero, zero, gravity_z);
                rotation_matrix_mid * self.accel + gravity_ned
            }
            AccelConvention::Coordinate => rotation_matrix_mid * self.accel,
        };

        // Propagate position and velocity
        let v_old = Vector3::new(ekf.state[IDX_VN], ekf.state[IDX_VE], ekf.state[IDX_VD]);
        let p_old = Vector3::new(ekf.state[IDX_N], ekf.state[IDX_E], ekf.state[IDX_D]);
        let p_new = p_old + v_old * dt + kinematic_accelaration * (half * dt * dt);
        let v_new = v_old + kinematic_accelaration * dt;
        ekf.state[IDX_N] = p_new.x;
        ekf.state[IDX_E] = p_new.y;
        ekf.state[IDX_D] = p_new.z;
        ekf.state[IDX_VN] = v_new.x;
        ekf.state[IDX_VE] = v_new.y;
        ekf.state[IDX_VD] = v_new.z;

        // Propagate quaternion
        let dq = UnitQuaternion::from_scaled_axis(self.gyro * dt);
        ekf.attitude *= dq;

        // Error-state attitude is not propagated (normal for MEKF)

        // Build the state transition matrix for covariance propagation.
        let mut state_transition = CovMatrix::identity();

        // Velocity integration into position
        for i in 0..3 {
            state_transition[(IDX_N + i, IDX_VN + i)] = dt;
        }

        // Attitude error coupling into position and velocity.
        let r_fskew = rotation_matrix * self.accel.cross_matrix();
        for i in 0..3 {
            for j in 0..3 {
                state_transition[(IDX_N + i, IDX_AX + j)] = -half * r_fskew[(i, j)] * dt * dt;
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                state_transition[(IDX_VN + i, IDX_AX + j)] = -r_fskew[(i, j)] * dt;
            }
        }

        // Gyro coupling into attitude error.
        let omega_skew = self.gyro.cross_matrix();
        for i in 0..3 {
            for j in 0..3 {
                state_transition[(IDX_AX + i, IDX_AX + j)] -= omega_skew[(i, j)] * dt;
            }
        }

        // P = F * P * Ft
        ekf.covariance = state_transition * ekf.covariance * state_transition.transpose();

        // Process noise (Discrete White Noise Acceleration on position/velocity, white-noise on attitude).
        let var_a = self.noise.accel_stddev * self.noise.accel_stddev;
        let var_g = self.noise.gyro_stddev * self.noise.gyro_stddev;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;
        for i in 0..3 {
            ekf.covariance[(IDX_N + i, IDX_N + i)] += var_a * dt4 / four;
            ekf.covariance[(IDX_N + i, IDX_VN + i)] += var_a * dt3 / two;
            ekf.covariance[(IDX_VN + i, IDX_N + i)] += var_a * dt3 / two;
            ekf.covariance[(IDX_VN + i, IDX_VN + i)] += var_a * dt2;
            ekf.covariance[(IDX_AX + i, IDX_AX + i)] += var_g * dt2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Ekf;
    use core::f64::consts::FRAC_PI_2;

    fn noise() -> ImuNoise<f64> {
        ImuNoise {
            accel_stddev: 0.1,
            gyro_stddev: 0.01,
        }
    }

    /// A stationary IMU (gyro = 0, accel = −g in body) should
    /// produce no position or velocity change and no attitude drift.
    #[test]
    fn stationary_specific_force_does_not_drift() {
        let mut ekf = Ekf::new();
        let mut imu = Imu::default()
            .set_accel(Vector3::new(0.0, 0.0, -9.81))
            .set_noise(noise())
            .set_accel_convention(AccelConvention::SpecificForce);
        for _ in 0..100 {
            ekf.predict(&mut imu, 0.01);
        }
        assert!(ekf.position().norm() < 1e-6);
        assert!(ekf.velocity().norm() < 1e-6);
        assert!(ekf.attitude.angle() < 1e-9);
    }

    /// A stationary IMU with coordinate-accel convention should also produce no position or velocity change
    #[test]
    fn stationary_coordinate_does_not_drift() {
        let mut ekf = Ekf::new();
        let mut imu = Imu::default()
            .set_noise(noise())
            .set_accel_convention(AccelConvention::Coordinate);
        for _ in 0..100 {
            ekf.predict(&mut imu, 0.01);
        }
        assert!(ekf.position().norm() < 1e-6);
        assert!(ekf.velocity().norm() < 1e-6);
    }

    /// With identity attitude, a north-pointing specific force (plus gravity
    /// cancellation on Z) should accelerate the body northward at 1 m/s^2.
    #[test]
    fn north_accel_produces_north_velocity() {
        let mut ekf = Ekf::new();
        let mut imu = Imu::default()
            .set_accel(Vector3::new(1.0, 0.0, -9.81))
            .set_noise(noise());
        let dt = 0.01;
        let steps = 100;
        for _ in 0..steps {
            ekf.predict(&mut imu, dt);
        }
        let t = dt * steps as f64; // 1 s
        assert!((ekf.velocity()[0] - 1.0 * t).abs() < 1e-9);
        assert!((ekf.position()[IDX_N] - 0.5 * 1.0 * t * t).abs() < 1e-9);
        assert!(ekf.velocity()[1].abs() < 1e-9);
        assert!(ekf.velocity()[2].abs() < 1e-9);
    }

    /// Gyro input rotates the nominal quaternion.
    #[test]
    fn gyro_rotates_attitude() {
        let mut ekf = Ekf::new();
        // 90deg/s about body-x for 1 s → 90deg roll rotation.
        let mut imu = Imu::default()
            .set_accel(Vector3::new(0.0, 0.0, -9.81))
            .set_gyro(Vector3::new(FRAC_PI_2, 0.0, 0.0))
            .set_noise(noise())
            .set_accel_convention(AccelConvention::SpecificForce);
        for _ in 0..1_000 {
            ekf.predict(&mut imu, 0.001);
        }
        // Angle about x should be ~PI/2.
        let axis_angle = ekf.attitude.scaled_axis();
        assert!(
            (axis_angle[0] - FRAC_PI_2).abs() < 1e-3,
            "got {:?}",
            axis_angle
        );
        assert!(axis_angle[1].abs() < 1e-3);
        assert!(axis_angle[2].abs() < 1e-3);
    }
}
