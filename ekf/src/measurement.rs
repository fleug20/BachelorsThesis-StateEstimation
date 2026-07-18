//! Extensibility hook for correction-step measurements.

use nalgebra::RealField;

use crate::{Ekf, JacobianRow};

/// A single scalar innovation, ready to be folded into the filter.
#[derive(Debug, Clone, Copy)]
pub struct ScalarUpdate<T: RealField + Copy = f64> {
    /// Row Jacobian `H`
    pub jacobian: JacobianRow<T>,
    /// Residual `y = z - h(x)`
    pub residual: T,
    /// Measurement-noise variance `R`
    pub variance: T,
}

/// A measurement that the EKF can consume in its correction step.
pub trait Measurement<T: RealField + Copy = f64> {
    const DIM: usize;

    /// Build the `i`-th scalar update against the current state.
    fn scalar(&mut self, i: usize, ekf: &Ekf<T>) -> Option<ScalarUpdate<T>>;

    /// Fill `out` with the residuals recorded during the last `correct()` call.
    ///
    /// Each entry is `Some(residual)` if that scalar was accepted, or `None` if
    /// the measurement rejected it (e.g. ToF with too steep a tilt).  `out` is
    /// filled up to `out.len().min(DIM)` entries; excess slots are left unchanged.
    ///
    /// The default implementation sets every slot to `None`, which is correct for
    /// any `Measurement` implementation that does not opt into `stats` storage.
    #[cfg(feature = "stats")]
    fn last_residuals(&self, out: &mut [Option<T>]) {
        for slot in out.iter_mut() {
            *slot = None;
        }
    }

    /// Fill `out` with the innovation covariances (`s = H P Hᵀ + R`) recorded
    /// during the last `correct()` call, one entry per scalar dimension.
    ///
    /// Each entry is `Some(s)` if that scalar was accepted, or `None` if it was
    /// rejected.  `out` is filled up to `out.len().min(DIM)` entries; excess
    /// slots are left unchanged.
    #[cfg(feature = "stats")]
    fn last_innovation_covariance(&self, out: &mut [Option<T>]) {
        for slot in out.iter_mut() {
            *slot = None;
        }
    }
}
