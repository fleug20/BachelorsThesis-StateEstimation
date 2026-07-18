//! Python bindings (pyo3). Enabled with `--features python`.
//!
//! Exposes an `ekf` Python module with:
//! - Filter state: `Ekf`
//! - Predictors: `ConstantVelocity`, `Imu`, `ImuNoise`, `AccelConvention`
//! - Measurements: `Gnss`, `GeodeticOrigin`, `UwbAnchor`, `Tof`, `AdaptiveTof`, `AdaptiveTofEma`
//!
//! Vectors and matrices cross the boundary as `numpy` `float64` arrays.
//! Attitude is exposed as a 4-element `[w, x, y, z]` quaternion.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::adaptive_measurement::{ExponentialDecay, Window};
use crate::measurements::{
    AdaptiveGnss, AdaptiveTof, AdaptiveUwbAnchor, GeodeticOrigin, Gnss, Tof, UwbAnchor,
};
use crate::predictors::{AccelConvention, ConstantVelocity, Imu, ImuNoise};
use crate::{
    CovMatrix, Ekf, IDX_AX, IDX_AY, IDX_AZ, IDX_D, IDX_E, IDX_N, IDX_VD, IDX_VE, IDX_VN,
    Measurement, STATE_DIM, StateVec,
};

fn vec3_from_array(arr: &PyReadonlyArray1<f64>) -> PyResult<Vector3<f64>> {
    let slice = arr.as_slice()?;
    if slice.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "expected a length-3 array, got length {}",
            slice.len()
        )));
    }
    Ok(Vector3::new(slice[0], slice[1], slice[2]))
}

#[pyclass(name = "ConstantVelocity", module = "ekf")]
#[derive(Clone, Copy)]
pub struct PyConstantVelocity {
    pub inner: ConstantVelocity,
}

#[pymethods]
impl PyConstantVelocity {
    #[new]
    fn new(accel_stddev: f64) -> Self {
        Self {
            inner: ConstantVelocity { accel_stddev },
        }
    }

    #[getter]
    fn accel_stddev(&self) -> f64 {
        self.inner.accel_stddev
    }

    fn __repr__(&self) -> String {
        format!("ConstantVelocity(accel_stddev={})", self.inner.accel_stddev)
    }
}

#[pyclass(name = "AccelConvention", module = "ekf", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyAccelConvention {
    SpecificForce,
    Coordinate,
}

impl PyAccelConvention {
    fn to_rust(self) -> AccelConvention {
        match self {
            Self::SpecificForce => AccelConvention::SpecificForce,
            Self::Coordinate => AccelConvention::Coordinate,
        }
    }
}

#[pyclass(name = "ImuNoise", module = "ekf")]
#[derive(Clone, Copy)]
pub struct PyImuNoise {
    pub inner: ImuNoise,
}

#[pymethods]
impl PyImuNoise {
    #[new]
    fn new(accel_stddev: f64, gyro_stddev: f64) -> Self {
        Self {
            inner: ImuNoise::new(accel_stddev, gyro_stddev),
        }
    }

    #[getter]
    fn accel_stddev(&self) -> f64 {
        self.inner.accel_stddev()
    }

    #[getter]
    fn gyro_stddev(&self) -> f64 {
        self.inner.gyro_stddev()
    }

    fn __repr__(&self) -> String {
        format!(
            "ImuNoise(accel_stddev={}, gyro_stddev={})",
            self.inner.accel_stddev(),
            self.inner.gyro_stddev(),
        )
    }
}

#[pyclass(name = "Imu", module = "ekf")]
#[derive(Clone)]
pub struct PyImu {
    accel: Vector3<f64>,
    gyro: Vector3<f64>,
    noise: ImuNoise,
    accel_convention: AccelConvention,
}

#[pymethods]
impl PyImu {
    #[new]
    fn new(
        accel: PyReadonlyArray1<f64>,
        gyro: PyReadonlyArray1<f64>,
        noise: PyImuNoise,
        accel_convention: PyAccelConvention,
    ) -> PyResult<Self> {
        Ok(Self {
            accel: vec3_from_array(&accel)?,
            gyro: vec3_from_array(&gyro)?,
            noise: noise.inner,
            accel_convention: accel_convention.to_rust(),
        })
    }

    #[getter]
    fn accel<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec![self.accel.x, self.accel.y, self.accel.z].into_pyarray(py)
    }

    #[getter]
    fn gyro<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec![self.gyro.x, self.gyro.y, self.gyro.z].into_pyarray(py)
    }

    #[getter]
    fn noise(&self) -> PyImuNoise {
        PyImuNoise { inner: self.noise }
    }
}

#[pyclass(name = "GeodeticOrigin", module = "ekf")]
#[derive(Clone, Copy)]
pub struct PyGeodeticOrigin {
    pub inner: GeodeticOrigin,
}

#[pymethods]
impl PyGeodeticOrigin {
    #[new]
    fn new(lat_deg: f64, lon_deg: f64, alt_m: f64) -> Self {
        Self {
            inner: GeodeticOrigin {
                lat_deg,
                lon_deg,
                alt_m,
            },
        }
    }

    #[getter]
    fn lat_deg(&self) -> f64 {
        self.inner.lat_deg
    }

    #[getter]
    fn lon_deg(&self) -> f64 {
        self.inner.lon_deg
    }

    #[getter]
    fn alt_m(&self) -> f64 {
        self.inner.alt_m
    }

    fn __repr__(&self) -> String {
        format!(
            "GeodeticOrigin(lat_deg={}, lon_deg={}, alt_m={})",
            self.inner.lat_deg, self.inner.lon_deg, self.inner.alt_m
        )
    }
}

#[pyclass(name = "Gnss", module = "ekf")]
#[derive(Clone, Copy)]
pub struct PyGnss {
    pub inner: Gnss,
}

#[pymethods]
impl PyGnss {
    #[new]
    fn new(ned: PyReadonlyArray1<f64>, stddev: PyReadonlyArray1<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Gnss::from_ned(vec3_from_array(&ned)?, vec3_from_array(&stddev)?),
        })
    }

    #[staticmethod]
    fn from_geodetic(
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        origin: &PyGeodeticOrigin,
        stddev: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Gnss::<f64>::from_geodetic(
                lat_deg,
                lon_deg,
                alt_m,
                &origin.inner,
                vec3_from_array(&stddev)?,
            ),
        })
    }

    #[getter]
    fn ned<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec![self.inner.ned.x, self.inner.ned.y, self.inner.ned.z].into_pyarray(py)
    }

    #[setter]
    fn set_ned(&mut self, value: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.inner.set_ned(vec3_from_array(&value)?);
        Ok(())
    }

    #[getter]
    fn stddev<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec![
            self.inner.stddev.x,
            self.inner.stddev.y,
            self.inner.stddev.z,
        ]
        .into_pyarray(py)
    }

    #[setter]
    fn set_stddev(&mut self, value: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.inner.stddev = vec3_from_array(&value)?;
        Ok(())
    }

    fn set_geodetic(&mut self, lat_deg: f64, lon_deg: f64, alt_m: f64, origin: &PyGeodeticOrigin) {
        self.inner
            .set_geodetic(lat_deg, lon_deg, alt_m, &origin.inner);
    }
}

const ADAPTIVE_GNSS_WINDOW: usize = 250;
const ADAPTIVE_UWB_WINDOW: usize = 250;
const ADAPTIVE_TOF_WINDOW: usize = 200;

#[pyclass(name = "AdaptiveGnss", module = "ekf")]
pub struct PyAdaptiveGnss {
    pub inner: AdaptiveGnss<Window<f64, ADAPTIVE_GNSS_WINDOW>>,
}

#[pymethods]
impl PyAdaptiveGnss {
    #[new]
    fn new(ned: PyReadonlyArray1<f64>, stddev: PyReadonlyArray1<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: AdaptiveGnss::new(Gnss::from_ned(
                vec3_from_array(&ned)?,
                vec3_from_array(&stddev)?,
            )),
        })
    }

    #[staticmethod]
    fn from_geodetic(
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        origin: &PyGeodeticOrigin,
        stddev: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: AdaptiveGnss::new(Gnss::<f64>::from_geodetic(
                lat_deg,
                lon_deg,
                alt_m,
                &origin.inner,
                vec3_from_array(&stddev)?,
            )),
        })
    }

    #[getter]
    fn ned<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let ned = self.inner.gnss().ned;
        vec![ned.x, ned.y, ned.z].into_pyarray(py)
    }

    #[setter]
    fn set_ned(&mut self, value: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.inner.gnss_mut().set_ned(vec3_from_array(&value)?);
        Ok(())
    }

    #[getter]
    fn stddev<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let s = self.inner.gnss().stddev;
        vec![s.x, s.y, s.z].into_pyarray(py)
    }

    #[setter]
    fn set_stddev(&mut self, value: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.inner.gnss_mut().stddev = vec3_from_array(&value)?;
        Ok(())
    }

    fn set_geodetic(&mut self, lat_deg: f64, lon_deg: f64, alt_m: f64, origin: &PyGeodeticOrigin) {
        self.inner
            .gnss_mut()
            .set_geodetic(lat_deg, lon_deg, alt_m, &origin.inner);
    }

    #[getter]
    fn current_variances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner
            .current_variances()
            .iter()
            .map(|opt| opt.unwrap_or(f64::NAN))
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }
}

#[pyclass(name = "AdaptiveGnssEma", module = "ekf")]
pub struct PyAdaptiveGnssEma {
    pub inner: AdaptiveGnss<ExponentialDecay<f64>>,
}

#[pymethods]
impl PyAdaptiveGnssEma {
    #[new]
    fn new(
        ned: PyReadonlyArray1<f64>,
        stddev: PyReadonlyArray1<f64>,
        alpha: f64,
        initial_var: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: AdaptiveGnss::new_ema(
                Gnss::from_ned(vec3_from_array(&ned)?, vec3_from_array(&stddev)?),
                alpha,
                initial_var,
            ),
        })
    }

    #[staticmethod]
    fn from_geodetic(
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        origin: &PyGeodeticOrigin,
        stddev: PyReadonlyArray1<f64>,
        alpha: f64,
        initial_var: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: AdaptiveGnss::new_ema(
                Gnss::<f64>::from_geodetic(
                    lat_deg,
                    lon_deg,
                    alt_m,
                    &origin.inner,
                    vec3_from_array(&stddev)?,
                ),
                alpha,
                initial_var,
            ),
        })
    }

    #[getter]
    fn ned<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let ned = self.inner.gnss().ned;
        vec![ned.x, ned.y, ned.z].into_pyarray(py)
    }

    #[setter]
    fn set_ned(&mut self, value: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.inner.gnss_mut().set_ned(vec3_from_array(&value)?);
        Ok(())
    }

    #[getter]
    fn stddev<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let s = self.inner.gnss().stddev;
        vec![s.x, s.y, s.z].into_pyarray(py)
    }

    #[setter]
    fn set_stddev(&mut self, value: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.inner.gnss_mut().stddev = vec3_from_array(&value)?;
        Ok(())
    }

    fn set_geodetic(&mut self, lat_deg: f64, lon_deg: f64, alt_m: f64, origin: &PyGeodeticOrigin) {
        self.inner
            .gnss_mut()
            .set_geodetic(lat_deg, lon_deg, alt_m, &origin.inner);
    }

    #[getter]
    fn current_variances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner
            .current_variances()
            .iter()
            .map(|opt| opt.unwrap_or(f64::NAN))
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }
}

#[pyclass(name = "UwbAnchor", module = "ekf")]
#[derive(Clone, Copy)]
pub struct PyUwbAnchor {
    pub inner: UwbAnchor,
}

#[pymethods]
impl PyUwbAnchor {
    #[new]
    fn new(ned: PyReadonlyArray1<f64>, distance: f64, stddev: f64) -> PyResult<Self> {
        Ok(Self {
            inner: UwbAnchor::new(vec3_from_array(&ned)?, distance, stddev),
        })
    }

    #[getter]
    fn ned<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec![self.inner.ned.x, self.inner.ned.y, self.inner.ned.z].into_pyarray(py)
    }

    #[getter]
    fn distance(&self) -> f64 {
        self.inner.distance
    }

    #[getter]
    fn stddev(&self) -> f64 {
        self.inner.stddev
    }

    fn __repr__(&self) -> String {
        format!(
            "UwbAnchor(ned=[{}, {}, {}], distance={}, stddev={})",
            self.inner.ned.x,
            self.inner.ned.y,
            self.inner.ned.z,
            self.inner.distance,
            self.inner.stddev,
        )
    }
}

#[pyclass(name = "AdaptiveUwbAnchor", module = "ekf")]
pub struct PyAdaptiveUwbAnchor {
    pub inner: AdaptiveUwbAnchor<Window<f64, ADAPTIVE_UWB_WINDOW>>,
}

#[pymethods]
impl PyAdaptiveUwbAnchor {
    #[new]
    fn new(ned: PyReadonlyArray1<f64>, distance: f64, stddev: f64) -> PyResult<Self> {
        Ok(Self {
            inner: AdaptiveUwbAnchor::new(UwbAnchor::new(vec3_from_array(&ned)?, distance, stddev)),
        })
    }

    #[getter]
    fn ned<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let ned = self.inner.anchor().ned;
        vec![ned.x, ned.y, ned.z].into_pyarray(py)
    }

    #[setter]
    fn set_ned(&mut self, value: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.inner.anchor_mut().ned = vec3_from_array(&value)?;
        Ok(())
    }

    #[getter]
    fn distance(&self) -> f64 {
        self.inner.anchor().distance
    }

    #[setter]
    fn set_distance(&mut self, value: f64) {
        self.inner.anchor_mut().distance = value;
    }

    #[getter]
    fn stddev(&self) -> f64 {
        self.inner.anchor().stddev
    }

    #[setter]
    fn set_stddev(&mut self, value: f64) {
        self.inner.anchor_mut().stddev = value;
    }

    #[getter]
    fn current_variance(&self) -> f64 {
        self.inner.current_variances()[0].unwrap_or(f64::NAN)
    }
}

#[pyclass(name = "AdaptiveUwbAnchorEma", module = "ekf")]
pub struct PyAdaptiveUwbAnchorEma {
    pub inner: AdaptiveUwbAnchor<ExponentialDecay<f64>>,
}

#[pymethods]
impl PyAdaptiveUwbAnchorEma {
    #[new]
    fn new(
        ned: PyReadonlyArray1<f64>,
        distance: f64,
        stddev: f64,
        alpha: f64,
        initial_var: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: AdaptiveUwbAnchor::new_ema(
                UwbAnchor::new(vec3_from_array(&ned)?, distance, stddev),
                alpha,
                initial_var,
            ),
        })
    }

    #[getter]
    fn ned<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let ned = self.inner.anchor().ned;
        vec![ned.x, ned.y, ned.z].into_pyarray(py)
    }

    #[setter]
    fn set_ned(&mut self, value: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.inner.anchor_mut().ned = vec3_from_array(&value)?;
        Ok(())
    }

    #[getter]
    fn distance(&self) -> f64 {
        self.inner.anchor().distance
    }

    #[setter]
    fn set_distance(&mut self, value: f64) {
        self.inner.anchor_mut().distance = value;
    }

    #[getter]
    fn stddev(&self) -> f64 {
        self.inner.anchor().stddev
    }

    #[setter]
    fn set_stddev(&mut self, value: f64) {
        self.inner.anchor_mut().stddev = value;
    }

    #[getter]
    fn current_variance(&self) -> f64 {
        self.inner.current_variances()[0].unwrap_or(f64::NAN)
    }
}

#[pyclass(name = "Tof", module = "ekf")]
#[derive(Clone, Copy)]
pub struct PyTof {
    pub inner: Tof,
}

#[pymethods]
impl PyTof {
    #[new]
    fn new(distance: f64, stddev: f64) -> Self {
        Self {
            inner: Tof::new(distance, stddev),
        }
    }

    #[getter]
    fn distance(&self) -> f64 {
        self.inner.distance
    }

    #[setter]
    fn set_distance(&mut self, value: f64) {
        self.inner.distance = value;
    }

    #[getter]
    fn stddev(&self) -> f64 {
        self.inner.stddev
    }

    #[setter]
    fn set_stddev(&mut self, value: f64) {
        self.inner.stddev = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "Tof(distance={}, stddev={})",
            self.inner.distance, self.inner.stddev,
        )
    }
}

#[pyclass(name = "AdaptiveTof", module = "ekf")]
pub struct PyAdaptiveTof {
    pub inner: AdaptiveTof<Window<f64, ADAPTIVE_TOF_WINDOW>>,
}

#[pymethods]
impl PyAdaptiveTof {
    #[new]
    fn new(distance: f64, stddev: f64) -> Self {
        Self {
            inner: AdaptiveTof::new(Tof::new(distance, stddev)),
        }
    }

    #[getter]
    fn distance(&self) -> f64 {
        self.inner.tof().distance
    }

    #[setter]
    fn set_distance(&mut self, value: f64) {
        self.inner.tof_mut().distance = value;
    }

    #[getter]
    fn stddev(&self) -> f64 {
        self.inner.tof().stddev
    }

    #[setter]
    fn set_stddev(&mut self, value: f64) {
        self.inner.tof_mut().stddev = value;
    }

    #[getter]
    fn current_variance(&self) -> f64 {
        self.inner.current_variances()[0].unwrap_or(f64::NAN)
    }
}

#[pyclass(name = "AdaptiveTofEma", module = "ekf")]
pub struct PyAdaptiveTofEma {
    pub inner: AdaptiveTof<ExponentialDecay<f64>>,
}

#[pymethods]
impl PyAdaptiveTofEma {
    #[new]
    fn new(distance: f64, stddev: f64, alpha: f64, initial_var: f64) -> Self {
        Self {
            inner: AdaptiveTof::new_ema(Tof::new(distance, stddev), alpha, initial_var),
        }
    }

    #[getter]
    fn distance(&self) -> f64 {
        self.inner.tof().distance
    }

    #[setter]
    fn set_distance(&mut self, value: f64) {
        self.inner.tof_mut().distance = value;
    }

    #[getter]
    fn stddev(&self) -> f64 {
        self.inner.tof().stddev
    }

    #[setter]
    fn set_stddev(&mut self, value: f64) {
        self.inner.tof_mut().stddev = value;
    }

    #[getter]
    fn current_variance(&self) -> f64 {
        self.inner.current_variances()[0].unwrap_or(f64::NAN)
    }
}

#[pyclass(name = "Ekf", module = "ekf")]
#[derive(Clone)]
pub struct PyEkf {
    pub inner: Ekf,
}

#[pymethods]
impl PyEkf {
    #[new]
    fn new() -> Self {
        Self { inner: Ekf::new() }
    }

    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner
            .state
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    #[setter]
    fn set_state(&mut self, value: PyReadonlyArray1<f64>) -> PyResult<()> {
        let slice = value.as_slice()?;
        if slice.len() != STATE_DIM {
            return Err(PyValueError::new_err(format!(
                "state must have length {STATE_DIM}, got {}",
                slice.len()
            )));
        }
        self.inner.state = StateVec::from_column_slice(slice);
        Ok(())
    }

    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut data = Vec::with_capacity(STATE_DIM * STATE_DIM);
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                data.push(self.inner.covariance[(i, j)]);
            }
        }
        data.into_pyarray(py)
            .reshape([STATE_DIM, STATE_DIM])
            .map_err(Into::into)
    }

    #[setter]
    fn set_covariance(&mut self, value: PyReadonlyArray2<f64>) -> PyResult<()> {
        let shape = value.shape();
        if shape != [STATE_DIM, STATE_DIM] {
            return Err(PyValueError::new_err(format!(
                "covariance must have shape ({STATE_DIM}, {STATE_DIM}), got {:?}",
                shape
            )));
        }
        let mut cov = CovMatrix::zeros();
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                cov[(i, j)] = *value.get([i, j]).unwrap();
            }
        }
        self.inner.covariance = cov;
        Ok(())
    }

    /// Quaternion as `[w, x, y, z]`, rotating body-frame vectors into NED.
    #[getter]
    fn attitude<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let q = self.inner.attitude.into_inner();
        vec![q.w, q.i, q.j, q.k].into_pyarray(py)
    }

    #[setter]
    fn set_attitude(&mut self, value: PyReadonlyArray1<f64>) -> PyResult<()> {
        let slice = value.as_slice()?;
        if slice.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "attitude must have length 4 ([w, x, y, z]), got {}",
                slice.len()
            )));
        }
        let q = Quaternion::new(slice[0], slice[1], slice[2], slice[3]);
        self.inner.attitude = UnitQuaternion::from_quaternion(q);
        Ok(())
    }

    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let p = self.inner.position();
        vec![p.x, p.y, p.z].into_pyarray(py)
    }

    #[getter]
    fn velocity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let v = self.inner.velocity();
        vec![v.x, v.y, v.z].into_pyarray(py)
    }

    /// Propagate forward by `dt` seconds using one of the supported predictors.
    fn predict(&mut self, model: &Bound<'_, PyAny>, dt: f64) -> PyResult<()> {
        if let Ok(m) = model.extract::<PyRef<PyConstantVelocity>>() {
            let mut cv = m.inner;
            self.inner.predict(&mut cv, dt);
        } else if let Ok(mut m) = model.extract::<PyRefMut<PyImu>>() {
            let accel = m.accel;
            let gyro = m.gyro;
            let noise = m.noise;
            let accel_convention = m.accel_convention;
            let mut imu = Imu::default()
                .set_accel(accel)
                .set_gyro(gyro)
                .set_noise(noise)
                .set_accel_convention(accel_convention);
            self.inner.predict(&mut imu, dt);
        } else {
            return Err(PyValueError::new_err(
                "predict() expects a ConstantVelocity or Imu instance",
            ));
        }
        Ok(())
    }

    /// Fold a measurement into the filter.
    ///
    /// Returns a numpy array of the residuals recorded during this correction.
    /// Entries are the accepted scalar residuals in order; scalars rejected by
    /// the measurement (e.g. ToF with too steep a tilt) are omitted.
    fn correct<'py>(
        &mut self,
        py: Python<'py>,
        measurement: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut buf = [None::<f64>; 3];
        let dim = if let Ok(mut m) = measurement.extract::<PyRefMut<PyGnss>>() {
            self.inner.correct(&mut m.inner);
            m.inner.last_residuals(&mut buf);
            3
        } else if let Ok(mut m) = measurement.extract::<PyRefMut<PyAdaptiveGnss>>() {
            self.inner.correct(&mut m.inner);
            m.inner.last_residuals(&mut buf);
            3
        } else if let Ok(mut m) = measurement.extract::<PyRefMut<PyAdaptiveGnssEma>>() {
            self.inner.correct(&mut m.inner);
            m.inner.last_residuals(&mut buf);
            3
        } else if let Ok(mut m) = measurement.extract::<PyRefMut<PyUwbAnchor>>() {
            self.inner.correct(&mut m.inner);
            m.inner.last_residuals(&mut buf[..1]);
            1
        } else if let Ok(mut m) = measurement.extract::<PyRefMut<PyAdaptiveUwbAnchor>>() {
            self.inner.correct(&mut m.inner);
            m.inner.last_residuals(&mut buf[..1]);
            1
        } else if let Ok(mut m) = measurement.extract::<PyRefMut<PyAdaptiveUwbAnchorEma>>() {
            self.inner.correct(&mut m.inner);
            m.inner.last_residuals(&mut buf[..1]);
            1
        } else if let Ok(mut m) = measurement.extract::<PyRefMut<PyTof>>() {
            self.inner.correct(&mut m.inner);
            m.inner.last_residuals(&mut buf[..1]);
            1
        } else if let Ok(mut m) = measurement.extract::<PyRefMut<PyAdaptiveTof>>() {
            self.inner.correct(&mut m.inner);
            m.inner.last_residuals(&mut buf[..1]);
            1
        } else if let Ok(mut m) = measurement.extract::<PyRefMut<PyAdaptiveTofEma>>() {
            self.inner.correct(&mut m.inner);
            m.inner.last_residuals(&mut buf[..1]);
            1
        } else {
            return Err(PyValueError::new_err(
                "correct() expects a Gnss, AdaptiveGnss, AdaptiveGnssEma, UwbAnchor, AdaptiveUwbAnchor, AdaptiveUwbAnchorEma, Tof, AdaptiveTof, or AdaptiveTofEma instance",
            ));
        };
        let residuals: Vec<f64> = buf[..dim].iter().filter_map(|&v| v).collect();
        Ok(residuals.into_pyarray(py))
    }

    fn finalize(&mut self) {
        self.inner.finalize();
    }
}

#[pymodule]
fn ekf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEkf>()?;
    m.add_class::<PyConstantVelocity>()?;
    m.add_class::<PyAccelConvention>()?;
    m.add_class::<PyImuNoise>()?;
    m.add_class::<PyImu>()?;
    m.add_class::<PyGeodeticOrigin>()?;
    m.add_class::<PyGnss>()?;
    m.add_class::<PyAdaptiveGnss>()?;
    m.add_class::<PyAdaptiveGnssEma>()?;
    m.add_class::<PyUwbAnchor>()?;
    m.add_class::<PyAdaptiveUwbAnchor>()?;
    m.add_class::<PyAdaptiveUwbAnchorEma>()?;
    m.add_class::<PyTof>()?;
    m.add_class::<PyAdaptiveTof>()?;
    m.add_class::<PyAdaptiveTofEma>()?;
    m.add("STATE_DIM", STATE_DIM)?;
    m.add("IDX_N", IDX_N)?;
    m.add("IDX_E", IDX_E)?;
    m.add("IDX_D", IDX_D)?;
    m.add("IDX_VN", IDX_VN)?;
    m.add("IDX_VE", IDX_VE)?;
    m.add("IDX_VD", IDX_VD)?;
    m.add("IDX_AX", IDX_AX)?;
    m.add("IDX_AY", IDX_AY)?;
    m.add("IDX_AZ", IDX_AZ)?;
    Ok(())
}
