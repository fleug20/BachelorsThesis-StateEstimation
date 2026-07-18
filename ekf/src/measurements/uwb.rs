//! UWB anchor distance measurement.
//!
//! Mirrors `mm_distance.c` from the Crazyflie firmware, translated to the
//! filter's local NED frame.

use nalgebra::{RealField, Vector3};

use crate::measurement::{Measurement, ScalarUpdate};
use crate::{Ekf, cast};
use crate::{IDX_D, IDX_E, IDX_N, JacobianRow};

/// A single UWB two-way-ranging measurement to a known anchor.
///
/// Both the anchor position and the filter state must be expressed in the same
/// local NED frame.
#[derive(Debug, Clone, Copy)]
pub struct UwbAnchor<T: RealField + Copy = f64> {
    /// Anchor position in the local NED frame [m].
    pub ned: Vector3<T>,
    /// Measured distance to the anchor [m].
    pub distance: T,
    /// 1-sigma measurement noise [m].
    pub stddev: T,
    #[cfg(feature = "stats")]
    pub residual: Option<T>,
    #[cfg(feature = "stats")]
    pub innovation_covariance: Option<T>,
}

impl<T: RealField + Copy> UwbAnchor<T> {
    pub fn new(ned: Vector3<T>, distance: T, stddev: T) -> Self {
        Self {
            ned,
            distance,
            stddev,
            #[cfg(feature = "stats")]
            residual: None,
            #[cfg(feature = "stats")]
            innovation_covariance: None,
        }
    }
}

impl<T: RealField + Copy> Measurement<T> for UwbAnchor<T> {
    const DIM: usize = 1;

    fn scalar(&mut self, _i: usize, ekf: &Ekf<T>) -> Option<ScalarUpdate<T>> {
        let dn = ekf.state[IDX_N] - self.ned[0];
        let de = ekf.state[IDX_E] - self.ned[1];
        let dd = ekf.state[IDX_D] - self.ned[2];

        let predicted = (dn * dn + de * de + dd * dd).sqrt();

        let zero: T = cast(0.0);
        let mut jacobian = JacobianRow::zeros();
        if predicted != zero {
            jacobian[(0, IDX_N)] = dn / predicted;
            jacobian[(0, IDX_E)] = de / predicted;
            jacobian[(0, IDX_D)] = dd / predicted;
        } else {
            jacobian[(0, IDX_N)] = cast(1.0);
        }

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
    fn anchor_directly_below_pulls_state_down() {
        // Anchor at D=5. Drone at D=0. Predicted distance is 5m.
        // Drone is further down -> D increases
        let mut ekf = Ekf::new();
        let mut anchor = UwbAnchor::new(Vector3::new(0.0, 0.0, 5.0), 0.0, 0.5);
        ekf.correct(&mut anchor);
        assert!(
            ekf.position()[IDX_D] > 0.0,
            "D should increase (move down) toward anchor"
        );
    }

    #[test]
    fn known_position_residual_is_zero() {
        // Anchor and Drone at N=10. Predicted distance is 0m.
        // N is unchanged
        let mut ekf = Ekf::new();
        ekf.state[IDX_N] = 10.0;
        let mut anchor = UwbAnchor::new(Vector3::zeros(), 10.0, 0.1);
        let before = ekf.position();
        ekf.correct(&mut anchor);
        let after = ekf.position();
        assert!(
            (after - before).norm() < 1e-4,
            "state should not change with zero residual"
        );
    }

    #[test]
    fn repeated_corrections_shrink_variance() {
        let mut ekf = Ekf::new();
        ekf.state[IDX_N] = 5.0;
        let mut anchor = UwbAnchor::new(Vector3::zeros(), 5.0, 0.3);
        let initial_var = ekf.covariance[(IDX_N, IDX_N)];
        for _ in 0..20 {
            ekf.correct(&mut anchor);
        }
        assert!(ekf.covariance[(IDX_N, IDX_N)] < initial_var);
        assert!(ekf.covariance[(IDX_N, IDX_N)] > 0.0);
    }
}
