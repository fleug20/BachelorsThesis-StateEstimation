//! Time-of-flight height measurement.
//!
//! Mirrors `mm_tof.c` from the Crazyflie firmware, translated to the
//! filter's local NED frame.

use nalgebra::RealField;

use crate::measurement::{Measurement, ScalarUpdate};
use crate::{Ekf, cast};
use crate::{IDX_D, JacobianRow};

/// A single time-of-flight range-to-ground measurement.
///
/// The sensor points downward in the body frame. The ground is assumed to be
/// at D = 0 in the local NED frame (i.e. the NED origin is at ground level).
#[derive(Debug, Clone, Copy)]
pub struct Tof<T: RealField + Copy = f64> {
    /// Measured distance to the ground [m].
    pub distance: T,
    /// 1-sigma measurement noise [m].
    pub stddev: T,
    #[cfg(feature = "stats")]
    pub residual: Option<T>,
    #[cfg(feature = "stats")]
    pub innovation_covariance: Option<T>,
}

impl<T: RealField + Copy> Tof<T> {
    /// Returns `None` when the tilt is too steep for a reliable measurement
    /// (`r_zz <= 0` or `|r_zz| <= 0.1`), matching the guard in `mm_tof.c`.
    pub fn new(distance: T, stddev: T) -> Self {
        Self {
            distance,
            stddev,
            #[cfg(feature = "stats")]
            residual: None,
            #[cfg(feature = "stats")]
            innovation_covariance: None,
        }
    }
}

impl<T: RealField + Copy> Measurement<T> for Tof<T> {
    const DIM: usize = 1;

    fn scalar(&mut self, _i: usize, ekf: &Ekf<T>) -> Option<ScalarUpdate<T>> {
        #[cfg(feature = "stats")]
        {
            self.residual = None;
            self.innovation_covariance = None;
        }

        let r_zz = ekf.rotation_matrix()[(2, 2)];

        if r_zz.abs() > cast(0.1) && r_zz > cast(0.0) {
            let zero: T = cast(0.0);

            // Subtract half the sensor cone angle (7.5°), clamped to zero, matching mm_tof.c.
            let half_cone: T = cast(15.0_f64.to_radians() / 2.0);
            let mut cos_arg = r_zz;
            if cos_arg > cast(1.0) {
                cos_arg = cast(1.0);
            } // avoid floating point errors
            if cos_arg < cast(-1.0) {
                cos_arg = cast(-1.0);
            } // avoid floating point errors
            let tilt_raw = cos_arg.acos().abs() - half_cone;
            let tilt = if tilt_raw < zero { zero } else { tilt_raw };
            let cos_tilt = tilt.cos();

            // In NED the drone height above the D=0 ground plane is -state[IDX_D].
            let predicted = -ekf.state[IDX_D] / cos_tilt;

            let mut jacobian = JacobianRow::zeros();
            jacobian[(0, IDX_D)] = -cast::<T>(1.0) / cos_tilt;

            let residual = self.distance - predicted;
            let variance = self.stddev * self.stddev;
            #[cfg(feature = "stats")]
            {
                self.residual = Some(residual);
                self.innovation_covariance =
                    Some((jacobian * ekf.covariance * jacobian.transpose())[(0, 0)] + variance);
            }
            Some(ScalarUpdate {
                jacobian,
                residual,
                variance,
            })
        } else {
            None
        }
    }

    #[cfg(feature = "stats")]
    fn last_residuals(&self, out: &mut [Option<T>]) {
        if let Some(slot) = out.first_mut() {
            *slot = self.residual;
        }
    }

    #[cfg(feature = "stats")]
    fn last_innovation_covariance(&self, out: &mut [Option<T>]) {
        if let Some(slot) = out.first_mut() {
            *slot = self.innovation_covariance;
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::Ekf;

    #[test]
    fn level_flight_pulls_d_up_toward_measured_height() {
        // Drone at D=0 (ground), ToF reads 1m -> filter should push D negative (upward)
        let mut ekf = Ekf::new();
        let mut tof = Tof::new(1.0, 0.05);
        ekf.correct(&mut tof);
        assert!(
            ekf.position()[IDX_D] < 0.0,
            "D should decrease (move up) to match 1m height"
        );
    }

    #[test]
    fn known_height_residual_is_zero() {
        // Drone at D=-1 (1m above ground), ToF reads 1m -> no state change
        let mut ekf = Ekf::new();
        ekf.state[IDX_D] = -1.0;
        let mut tof = Tof::new(1.0, 0.05);
        let before = ekf.position();
        ekf.correct(&mut tof);
        let after = ekf.position();
        assert!(
            (after - before).norm() < 1e-4,
            "state should not change with zero residual"
        );
    }

    #[test]
    fn repeated_corrections_shrink_variance() {
        let mut ekf = Ekf::new();
        ekf.state[IDX_D] = -2.0;
        let mut tof = Tof::new(2.0, 0.1);
        let initial_var = ekf.covariance[(IDX_D, IDX_D)];
        for _ in 0..20 {
            ekf.correct(&mut tof);
        }
        assert!(ekf.covariance[(IDX_D, IDX_D)] < initial_var);
        assert!(ekf.covariance[(IDX_D, IDX_D)] > 0.0);
    }
}
