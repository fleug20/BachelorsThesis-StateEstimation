//! Extensibility hook for motion (prediction) models.
//!
//! To add a new model, create a struct in `crate::predictors`

use nalgebra::RealField;

use crate::Ekf;

/// A motion model that can be used in the EKF's prediction step.
pub trait Predictor<T: RealField + Copy = f64> {
    /// Propagate state, covariance, and attitude over `dt` seconds.
    fn apply(&mut self, ekf: &mut Ekf<T>, dt: T);
}
