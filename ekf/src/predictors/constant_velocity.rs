//! Constant-velocity motion model.
//!
//! Advances position by `v * dt`, leaves velocity unchanged, and does not touch
//! the attitude quaternion or attitude-error state. Process noise is the
//! discrete white-noise acceleration block (DWNA), applied per axis.

use nalgebra::RealField;

use crate::cast;
use crate::predict::Predictor;
use crate::{CovMatrix, Ekf};

#[derive(Debug, Clone, Copy)]
pub struct ConstantVelocity<T: RealField + Copy = f64> {
    /// Standard deviation of unmodelled acceleration [m/s^2].
    pub accel_stddev: T,
}

impl<T: RealField + Copy> Predictor<T> for ConstantVelocity<T> {
    fn apply(&mut self, ekf: &mut Ekf<T>, dt: T) {
        for i in 0..3 {
            let v = ekf.state[i + 3];
            ekf.state[i] += v * dt;
        }

        let two: T = cast(2.0);
        let four: T = cast(4.0);

        let mut state_transition = CovMatrix::identity();
        for i in 0..3 {
            state_transition[(i, i + 3)] = dt;
        }
        ekf.covariance = state_transition * ekf.covariance * state_transition.transpose();

        // DWNA process noise on position/velocity, per axis.
        let var_a = self.accel_stddev * self.accel_stddev;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;
        for i in 0..3 {
            ekf.covariance[(i, i)] += var_a * dt4 / four;
            ekf.covariance[(i, i + 3)] += var_a * dt3 / two;
            ekf.covariance[(i + 3, i)] += var_a * dt3 / two;
            ekf.covariance[(i + 3, i + 3)] += var_a * dt2;
        }
    }
}
