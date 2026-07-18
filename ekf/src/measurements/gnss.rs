//! GNSS position measurement, in WGS84 geodetic or local NED coordinates.

#[cfg(feature = "stats")]
use nalgebra::SMatrix;
use nalgebra::{RealField, Vector3};

use crate::measurement::{Measurement, ScalarUpdate};
use crate::{Ekf, cast};
use crate::{IDX_D, IDX_E, IDX_N, JacobianRow};

/// WGS84 geodetic reference point that anchors the filter's local NED frame.
#[derive(Debug, Clone, Copy)]
pub struct GeodeticOrigin<T: RealField + Copy = f64> {
    pub lat_deg: T,
    pub lon_deg: T,
    pub alt_m: T,
}

/// A position fix expressed in the filter's local NED frame.
///
/// Construct with [`Gnss::from_ned`] if you already have NED coordinates, or
/// [`Gnss::from_geodetic`] to convert a WGS84 lat/lon/alt against a known
/// origin.
#[derive(Debug, Clone, Copy)]
pub struct Gnss<T: RealField + Copy = f64> {
    /// Measurement in the local NED frame [m].
    pub ned: Vector3<T>,
    /// 1-sigma measurement noise per NED axis [m].
    pub stddev: Vector3<T>,
    #[cfg(feature = "stats")]
    pub residuals: [Option<T>; 3],
    #[cfg(feature = "stats")]
    pub innovation_covariances: [Option<T>; 3],
}

impl<T: RealField + Copy> Gnss<T> {
    pub fn from_ned(ned: Vector3<T>, stddev: Vector3<T>) -> Self {
        Self {
            ned,
            stddev,
            #[cfg(feature = "stats")]
            residuals: [None; 3],
            #[cfg(feature = "stats")]
            innovation_covariances: [None; 3],
        }
    }

    pub fn set_ned(&mut self, ned: Vector3<T>) {
        self.ned = ned;
    }

    fn geodetic_to_ned(
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        origin: &GeodeticOrigin<f64>,
    ) -> Vector3<f64> {
        let (north, east, down) = map_3d::geodetic2ned(
            lat_deg.to_radians(),
            lon_deg.to_radians(),
            alt_m,
            origin.lat_deg.to_radians(),
            origin.lon_deg.to_radians(),
            origin.alt_m,
            map_3d::Ellipsoid::WGS84,
        );

        Vector3::new(north, east, down)
    }
}

impl Gnss<f64> {
    /// Convert a WGS84 geodetic fix to a local NED fix and construct the
    /// measurement.
    ///
    /// The `map_3d` crate operates in `f64`, so this constructor is only
    /// available on `Gnss<f64>`.
    pub fn from_geodetic(
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        origin: &GeodeticOrigin<f64>,
        stddev: Vector3<f64>,
    ) -> Self {
        Self::from_ned(
            Self::geodetic_to_ned(lat_deg, lon_deg, alt_m, origin),
            stddev,
        )
    }

    pub fn set_geodetic(
        &mut self,
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        origin: &GeodeticOrigin<f64>,
    ) {
        self.set_ned(Self::geodetic_to_ned(lat_deg, lon_deg, alt_m, origin));
    }
}

impl Gnss<f32> {
    /// Convert a WGS84 geodetic fix to a local NED fix and construct the
    /// measurement.
    ///
    /// The `map_3d` crate operates in `f64`, so this constructor is only
    /// available on `Gnss<f64>`.
    pub fn from_geodetic(
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        origin: &GeodeticOrigin<f64>,
        stddev: Vector3<f32>,
    ) -> Self {
        Self::from_ned(
            nalgebra::convert(Self::geodetic_to_ned(lat_deg, lon_deg, alt_m, origin)),
            stddev,
        )
    }

    pub fn set_geodetic(
        &mut self,
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        origin: &GeodeticOrigin<f64>,
    ) {
        self.set_ned(nalgebra::convert(Self::geodetic_to_ned(
            lat_deg, lon_deg, alt_m, origin,
        )));
    }

    pub fn set_from_geodetic() {}
}

impl<T: RealField + Copy> Measurement<T> for Gnss<T> {
    const DIM: usize = 3;

    fn scalar(&mut self, i: usize, ekf: &Ekf<T>) -> Option<ScalarUpdate<T>> {
        let axis = [IDX_N, IDX_E, IDX_D][i];
        let mut jacobian = JacobianRow::zeros();
        jacobian[(0, axis)] = cast(1.0);
        let residual = self.ned[i] - ekf.state[axis];
        let variance = self.stddev[i] * self.stddev[i];
        #[cfg(feature = "stats")]
        {
            self.residuals[i] = Some(residual);
            let hpht: SMatrix<T, 1, 1> = jacobian * ekf.covariance * jacobian.transpose();
            self.innovation_covariances[i] = Some(hpht[(0, 0)] + variance);
        }
        Some(ScalarUpdate {
            jacobian,
            residual,
            variance,
        })
    }

    #[cfg(feature = "stats")]
    fn last_residuals(&self, out: &mut [Option<T>]) {
        for (slot, &src) in out.iter_mut().zip(self.residuals.iter()) {
            *slot = src;
        }
    }

    #[cfg(feature = "stats")]
    fn last_innovation_covariance(&self, out: &mut [Option<T>]) {
        for (slot, &src) in out.iter_mut().zip(self.innovation_covariances.iter()) {
            *slot = src;
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    #[cfg(feature = "std")]
    use crate::predictors::ConstantVelocity;
    use crate::{Ekf, IDX_VN};

    fn origin() -> GeodeticOrigin {
        GeodeticOrigin {
            lat_deg: 48.2082,
            lon_deg: 16.3738,
            alt_m: 170.0,
        }
    }

    #[test]
    fn gnss_at_origin_pulls_state_to_zero() {
        let mut ekf = Ekf::new();
        let o = origin();
        let mut m = Gnss::<f64>::from_geodetic(
            o.lat_deg,
            o.lon_deg,
            o.alt_m,
            &o,
            Vector3::new(1.0, 1.0, 2.0),
        );
        ekf.correct(&mut m);
        assert!(ekf.position().norm() < 0.1);
    }

    #[test]
    fn one_metre_north_produces_positive_n() {
        let mut ekf = Ekf::new();
        let o = origin();
        let one_m_in_deg = 1.0 / 111_320.0;
        let mut m = Gnss::<f64>::from_geodetic(
            o.lat_deg + one_m_in_deg,
            o.lon_deg,
            o.alt_m,
            &o,
            Vector3::new(0.5, 0.5, 1.0),
        );
        ekf.correct(&mut m);
        let p = ekf.position();
        assert!(
            (p[IDX_N] - 1.0).abs() < 0.2,
            "north should be ~1 m, got {}",
            p[IDX_N]
        );
        assert!(
            p[IDX_E].abs() < 0.2,
            "east should be ~0 m, got {}",
            p[IDX_E]
        );
    }

    #[test]
    fn altitude_above_origin_is_negative_d() {
        let mut ekf = Ekf::new();
        let o = origin();
        let mut m = Gnss::<f64>::from_geodetic(
            o.lat_deg,
            o.lon_deg,
            o.alt_m + 5.0,
            &o,
            Vector3::new(0.5, 0.5, 0.5),
        );
        ekf.correct(&mut m);
        assert!((ekf.position()[IDX_D] - (-5.0)).abs() < 0.2);
    }

    #[test]
    fn repeated_fixes_shrink_position_covariance() {
        let mut ekf = Ekf::new();
        let o = origin();
        let mut m = Gnss::<f64>::from_geodetic(
            o.lat_deg,
            o.lon_deg,
            o.alt_m,
            &o,
            Vector3::new(1.0, 1.0, 2.0),
        );
        let initial_var = ekf.covariance[(IDX_N, IDX_N)];
        for _ in 0..20 {
            ekf.correct(&mut m);
        }
        let final_var = ekf.covariance[(IDX_N, IDX_N)];
        assert!(final_var < initial_var);
        assert!(final_var > 0.0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn predict_then_gnss_converges_toward_fix() {
        let mut ekf = Ekf::new();
        let o = origin();
        ekf.state[IDX_VN] = 5.0;
        let mut cv = ConstantVelocity { accel_stddev: 0.5 };

        for _ in 0..50 {
            ekf.predict(&mut cv, 0.1);
            let mut m = Gnss::<f64>::from_geodetic(
                o.lat_deg,
                o.lon_deg,
                o.alt_m,
                &o,
                Vector3::new(0.5, 0.5, 1.0),
            );
            ekf.correct(&mut m);
        }
        assert!(ekf.position().norm() < 1.0);
    }
}
