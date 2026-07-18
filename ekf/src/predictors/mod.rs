//! Individual motion models
//!
//! Each submodule defines a struct implementing [`crate::predict::Predictor`].

#[cfg(feature = "std")]
pub mod constant_velocity;

pub mod imu;

#[cfg(feature = "std")]
pub use constant_velocity::ConstantVelocity;

pub use imu::{AccelConvention, Imu, ImuNoise};
