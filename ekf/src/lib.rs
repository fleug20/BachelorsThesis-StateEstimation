//! 9-state Multiplicative EKF for local-NED navigation.
//!
//! Fuses pluggable [`Predictor`] motion models with scalar [`Measurement`] corrections.
//! Attitude is carried as a [`nalgebra::UnitQuaternion`] with a small-angle error state.

#![cfg_attr(feature = "no_std", no_std)]

pub mod adaptive_measurement;
pub mod measurement;
pub mod measurements;
pub mod predict;
pub mod predictors;

pub use nalgebra;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "c_api")]
pub mod c_api;

pub use measurement::{Measurement, ScalarUpdate};
pub use predict::Predictor;

use nalgebra::{Matrix3, RealField, SMatrix, SVector, UnitQuaternion, Vector3};

pub const STATE_DIM: usize = 9;

pub type StateVec<T = f64> = SVector<T, STATE_DIM>;
pub type CovMatrix<T = f64> = SMatrix<T, STATE_DIM, STATE_DIM>;
pub type JacobianRow<T = f64> = SMatrix<T, 1, STATE_DIM>;

pub const IDX_N: usize = 0;
pub const IDX_E: usize = 1;
pub const IDX_D: usize = 2;
pub const IDX_VN: usize = 3;
pub const IDX_VE: usize = 4;
pub const IDX_VD: usize = 5;
pub const IDX_AX: usize = 6;
pub const IDX_AY: usize = 7;
pub const IDX_AZ: usize = 8;

pub const COVARIANCE_DIM: usize = STATE_DIM * STATE_DIM;

/// Initial diagonal variance for position states.
pub const MAX_COVARIANCE: f64 = 200.0;
/// Minimum allowed diagonal variance (prevents singular covariance).
pub const MIN_COVARIANCE: f64 = 1e-6;

/// Convert an `f64` literal to `T`. Uses fully-qualified syntax so no `use`
/// import is needed at the call site.
#[inline(always)]
pub(crate) fn cast<T: RealField>(x: f64) -> T {
    num_traits::FromPrimitive::from_f64(x).expect("f64 literal is representable as T")
}

/// 9-state MEKF in a local NED frame
///
/// State layout: `[N, E, D, VN, VE, VD, AX, AY, AZ]` where `AX..AZ` is the
/// small-angle attitude-error vector in body-frame axes. The nominal attitude
/// is held separately as a [`nalgebra::UnitQuaternion`] (body to NED).
///
/// The motion model is pluggable via [`Predictor`] (see `predictors/`) and the
/// correction step via [`Measurement`] (see `measurements/`).
///
/// The filter is generic over the float type `T` (defaulting to `f64`).
/// Use `Ekf<f32>` or `Ekf<f64>` explicitly, or rely on the default.
#[derive(Debug, Clone)]
pub struct Ekf<T: RealField + Copy = f64> {
    pub state: StateVec<T>,
    pub covariance: CovMatrix<T>,
    /// Rotation from body frame to NED.
    pub attitude: UnitQuaternion<T>,
}

#[cfg(not(feature = "no_std"))]
impl<T: RealField + Copy> Default for Ekf<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealField + Copy> Ekf<T> {
    #[cfg(not(feature = "no_std"))]
    pub fn new() -> Self {
        let mut ekf = Self {
            state: StateVec::zeros(),
            covariance: CovMatrix::zeros(),
            attitude: UnitQuaternion::identity(),
        };
        ekf.init();
        ekf
    }

    /// Reset this filter to its initial state without allocating a temporary on
    /// the stack.
    ///
    /// Equivalent to `*self = Ekf::new()` but writes directly into the existing
    /// allocation.  Use this when `self` lives in a static or a fixed-size buffer
    /// where stack space is limited.
    pub fn init(&mut self) {
        unsafe {
            // Zero state and covariance through their existing memory locations —
            // no full-struct stack temporary is created.
            core::ptr::write_bytes(self.state.as_mut_ptr(), 0, STATE_DIM);
            core::ptr::write_bytes(self.covariance.as_mut_ptr(), 0, COVARIANCE_DIM);
        }
        // Initial covariance diagonal
        for i in IDX_N..=IDX_D {
            self.covariance[(i, i)] = cast(MAX_COVARIANCE);
        }
        for i in IDX_VN..=IDX_VD {
            self.covariance[(i, i)] = cast(10.0);
        }
        for i in IDX_AX..=IDX_AZ {
            self.covariance[(i, i)] = cast(1.0);
        }
        self.attitude = UnitQuaternion::identity();
    }

    pub fn position(&self) -> Vector3<T> {
        Vector3::new(self.state[IDX_N], self.state[IDX_E], self.state[IDX_D])
    }

    pub fn velocity(&self) -> Vector3<T> {
        Vector3::new(self.state[IDX_VN], self.state[IDX_VE], self.state[IDX_VD])
    }

    pub fn rotation_matrix(&self) -> Matrix3<T> {
        self.attitude.to_rotation_matrix().into_inner()
    }

    /// Propagate forward by `dt` seconds using the chosen motion model.
    pub fn predict<P: Predictor<T> + ?Sized>(&mut self, model: &mut P, dt: T) {
        model.apply(self, dt);
        self.enforce_symmetry();
    }

    /// Fold a measurement into the filter and reset the error state.
    pub fn correct<M: Measurement<T> + ?Sized>(&mut self, measurement: &mut M) {
        for i in 0..M::DIM {
            let Some(update) = measurement.scalar(i, self) else {
                continue;
            };
            self.apply_scalar_update(update);
        }
        self.finalize();
    }

    /// This is specific for the MEKF:
    ///
    /// Fold the attitude-error state into the nominal quaternion and zero it.
    ///
    /// Automatically called by [`Ekf::correct`]. A first-order covariance reset
    /// (`P ← G P Gᵀ`) is skipped here - the correction it applies is identity
    /// to first order in the error state.
    pub fn finalize(&mut self) {
        let err = Vector3::new(self.state[IDX_AX], self.state[IDX_AY], self.state[IDX_AZ]);
        let dq = UnitQuaternion::from_scaled_axis(err);
        self.attitude *= dq;
        self.state[IDX_AX] = cast(0.0);
        self.state[IDX_AY] = cast(0.0);
        self.state[IDX_AZ] = cast(0.0);
    }

    fn apply_scalar_update(&mut self, update: ScalarUpdate<T>) {
        let ScalarUpdate {
            jacobian: h,
            residual,
            variance: r,
        } = update;

        let s = (h * self.covariance * h.transpose())[(0, 0)] + r;
        let k = self.covariance * h.transpose() / s;

        self.state += k * residual;

        // Joseph form
        let i_kh = CovMatrix::identity() - k * h;
        self.covariance = i_kh * self.covariance * i_kh.transpose() + (k * k.transpose()) * r;

        self.enforce_symmetry();
    }

    pub(crate) fn enforce_symmetry(&mut self) {
        let half: T = cast(0.5);
        for i in 0..STATE_DIM {
            for j in i..STATE_DIM {
                let v = half * (self.covariance[(i, j)] + self.covariance[(j, i)]);
                self.covariance[(i, j)] = v;
                self.covariance[(j, i)] = v;
            }
        }
    }
}
