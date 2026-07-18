//! C API for the EKF.
//!
//! All values crossing the C boundary are `f32`.  Internally the filter runs
//! as `Ekf<f32>`.
//!
//! # Safety contract
//!
//! [`rkf_init`] **must** be called once before any other `rkf_*` function.

#[cfg(feature = "stats")]
use crate::Measurement;
#[cfg(not(feature = "c_api_sliding_window"))]
use crate::adaptive_measurement::ExponentialDecay;
#[cfg(feature = "c_api_sliding_window")]
use crate::adaptive_measurement::Window;
use crate::measurements::{AdaptiveTof, AdaptiveUwbAnchor, Tof, UwbAnchor};
use crate::predictors::{Imu, ImuNoise};
use crate::{Ekf, STATE_DIM};

const COVARIANCE_DIM: usize = crate::COVARIANCE_DIM;
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ptr;
use nalgebra::Vector3;

#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct RkfState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub attitude: [f32; 4],
    pub covariance: [f32; COVARIANCE_DIM],
}

/// Maximum number of UWB anchors tracked by the adaptive filter.
pub const RKF_MAX_UWB_ANCHORS: usize = 8;

#[cfg(feature = "c_api_sliding_window")]
const SLIDING_WINDOW_SIZE_TOF: usize = 105;
#[cfg(feature = "c_api_sliding_window")]
const SLIDING_WINDOW_SIZE_UWB: usize = 115;

#[cfg(feature = "c_api_sliding_window")]
type AdaptiveStrategyTofF32 = Window<f32, SLIDING_WINDOW_SIZE_TOF>;
#[cfg(not(feature = "c_api_sliding_window"))]
type AdaptiveStrategyTofF32 = ExponentialDecay<f32>;

#[cfg(feature = "c_api_sliding_window")]
type AdaptiveStrategyUwbF32 = Window<f32, SLIDING_WINDOW_SIZE_UWB>;
#[cfg(not(feature = "c_api_sliding_window"))]
type AdaptiveStrategyUwbF32 = ExponentialDecay<f32>;

type AdaptiveAnchor = AdaptiveUwbAnchor<AdaptiveStrategyUwbF32, f32>;
type AdaptiveTofSensor = AdaptiveTof<AdaptiveStrategyTofF32, f32>;

#[cfg(feature = "c_api_sliding_window")]
#[inline]
fn make_adaptive_anchor(inner: UwbAnchor<f32>, _alpha: f32, _initial_var: f32) -> AdaptiveAnchor {
    AdaptiveUwbAnchor::new(inner)
}
#[cfg(not(feature = "c_api_sliding_window"))]
#[inline]
fn make_adaptive_anchor(inner: UwbAnchor<f32>, alpha: f32, initial_var: f32) -> AdaptiveAnchor {
    AdaptiveUwbAnchor::new_ema(inner, alpha, initial_var)
}

#[cfg(feature = "c_api_sliding_window")]
#[inline]
fn make_adaptive_tof(inner: Tof<f32>, _alpha: f32, _initial_var: f32) -> AdaptiveTofSensor {
    AdaptiveTof::new(inner)
}
#[cfg(not(feature = "c_api_sliding_window"))]
#[inline]
fn make_adaptive_tof(inner: Tof<f32>, alpha: f32, initial_var: f32) -> AdaptiveTofSensor {
    AdaptiveTof::new_ema(inner, alpha, initial_var)
}

struct EkfCell(UnsafeCell<MaybeUninit<Ekf<f32>>>);
unsafe impl Sync for EkfCell {}

struct ImuCell(UnsafeCell<MaybeUninit<Imu<f32>>>);
unsafe impl Sync for ImuCell {}

struct AdaptiveAnchorsCell(UnsafeCell<MaybeUninit<[AdaptiveAnchor; RKF_MAX_UWB_ANCHORS]>>);
unsafe impl Sync for AdaptiveAnchorsCell {}

struct AdaptiveTofCell(UnsafeCell<MaybeUninit<AdaptiveTofSensor>>);
unsafe impl Sync for AdaptiveTofCell {}

static EKF: EkfCell = EkfCell(UnsafeCell::new(MaybeUninit::uninit()));
static IMU: ImuCell = ImuCell(UnsafeCell::new(MaybeUninit::uninit()));
static ADAPTIVE_ANCHORS: AdaptiveAnchorsCell =
    AdaptiveAnchorsCell(UnsafeCell::new(MaybeUninit::uninit()));
static ADAPTIVE_TOF: AdaptiveTofCell = AdaptiveTofCell(UnsafeCell::new(MaybeUninit::uninit()));

#[cfg(feature = "stats")]
struct F32Cell(UnsafeCell<f32>);
#[cfg(feature = "stats")]
unsafe impl Sync for F32Cell {}

/// Last residual from `rkf_update_uwb`. `NaN` until the first call.
#[cfg(feature = "stats")]
static UWB_RESIDUAL: F32Cell = F32Cell(UnsafeCell::new(f32::NAN));
/// Last residual from `rkf_update_tof`. `NaN` until the first call.
#[cfg(feature = "stats")]
static TOF_RESIDUAL: F32Cell = F32Cell(UnsafeCell::new(f32::NAN));
/// Last innovation covariance from `rkf_update_uwb`. `NaN` until the first call.
#[cfg(feature = "stats")]
static UWB_INNOVATION_COVARIANCE: F32Cell = F32Cell(UnsafeCell::new(f32::NAN));
/// Last innovation covariance from `rkf_update_tof`. `NaN` until the first call.
#[cfg(feature = "stats")]
static TOF_INNOVATION_COVARIANCE: F32Cell = F32Cell(UnsafeCell::new(f32::NAN));

/// Initialise the filter and the IMU scratch space.
///
/// Must be called exactly once before any other `rkf_*` function.
///
/// # Safety
///
/// Not thread-safe. Call from a single context before starting any tasks that
/// use the other `rkf_*` functions.
#[unsafe(no_mangle)]
pub extern "C" fn rkf_init() {
    unsafe {
        // Zero the raw storage. All-zero is a valid bit pattern for both types:
        // every field is an f32 array (0x00…0 → 0.0f32) or an enum whose first
        // (default) variant has discriminant 0.
        let ekf_ptr = (*EKF.0.get()).as_mut_ptr();
        ptr::write_bytes(ekf_ptr, 0, 1);

        let imu_ptr = (*IMU.0.get()).as_mut_ptr();
        ptr::write_bytes(imu_ptr, 0, 1);

        // Fix up the non-zero EKF initial values in place. State is already
        // zero; covariance diagonal and the identity quaternion are set here.
        // AccelConvention::SpecificForce has discriminant 0, so the IMU needs
        // no further fixup.
        (*ekf_ptr).init();

        (*imu_ptr).init();
    }
    rkf_init_uwb_adaptive(0.05, 0.18);
    rkf_init_tof_adaptive(0.05, 0.0055);
}

/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[unsafe(no_mangle)]
pub extern "C" fn rkf_predict_imu(
    ax: f32,
    ay: f32,
    az: f32,
    gx: f32,
    gy: f32,
    gz: f32,
    accel_std: f32,
    gyro_std: f32,
    dt: f32,
) {
    let ekf = unsafe { &mut *(*EKF.0.get()).as_mut_ptr() };
    let imu = unsafe { &mut *(*IMU.0.get()).as_mut_ptr() };
    imu.update_measurement(
        Vector3::new(ax, ay, az),
        Vector3::new(gx, gy, gz),
        ImuNoise::new(accel_std, gyro_std),
    );
    ekf.predict(imu, dt);
}

#[unsafe(no_mangle)]
pub extern "C" fn rkf_update_uwb(
    anchor_n: f32,
    anchor_e: f32,
    anchor_d: f32,
    distance: f32,
    stddev: f32,
) {
    let ekf = unsafe { &mut *(*EKF.0.get()).as_mut_ptr() };
    let mut anchor = UwbAnchor::new(Vector3::new(anchor_n, anchor_e, anchor_d), distance, stddev);
    ekf.correct(&mut anchor);
    #[cfg(feature = "stats")]
    {
        let mut buf = [None::<f32>; 1];
        anchor.last_residuals(&mut buf);
        unsafe { *UWB_RESIDUAL.0.get() = buf[0].unwrap_or(f32::NAN) };
        anchor.last_innovation_covariance(&mut buf);
        unsafe { *UWB_INNOVATION_COVARIANCE.0.get() = buf[0].unwrap_or(f32::NAN) };
    }
}

/// (Re-)initialise all adaptive UWB anchor states with the given EMA parameters.
///
/// Called automatically by [`rkf_init`] with `alpha = 0.1`, `initial_var = 1.0`.
/// Call again to override those defaults before the first [`rkf_update_uwb_adaptive`].
#[unsafe(no_mangle)]
pub extern "C" fn rkf_init_uwb_adaptive(alpha: f32, initial_var: f32) {
    let anchors_ptr = unsafe { (*ADAPTIVE_ANCHORS.0.get()).as_mut_ptr() as *mut AdaptiveAnchor };
    for i in 0..RKF_MAX_UWB_ANCHORS {
        unsafe {
            ptr::write(
                anchors_ptr.add(i),
                make_adaptive_anchor(
                    UwbAnchor::new(Vector3::zeros(), 0.0, 0.18),
                    alpha,
                    initial_var,
                ),
            );
        }
    }
}

/// Fold an adaptive UWB measurement into the filter.
///
/// `anchor_id` selects which adaptive state slot to use (0 …
/// `RKF_MAX_UWB_ANCHORS - 1`). The call is a no-op for out-of-range IDs.
/// The measurement noise variance is estimated automatically by the EMA
/// strategy; no `stddev` argument is needed.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[unsafe(no_mangle)]
pub extern "C" fn rkf_update_uwb_adaptive(
    anchor_id: u8,
    anchor_n: f32,
    anchor_e: f32,
    anchor_d: f32,
    distance: f32,
) {
    if anchor_id as usize >= RKF_MAX_UWB_ANCHORS {
        return;
    }
    let ekf = unsafe { &mut *(*EKF.0.get()).as_mut_ptr() };
    let anchors_ptr = unsafe { (*ADAPTIVE_ANCHORS.0.get()).as_mut_ptr() as *mut AdaptiveAnchor };
    let anchor = unsafe { &mut *anchors_ptr.add(anchor_id as usize) };
    let inner = anchor.anchor_mut();
    inner.ned = Vector3::new(anchor_n, anchor_e, anchor_d);
    inner.distance = distance;
    ekf.correct(anchor);
}

/// (Re-)initialise the adaptive ToF state with the given EMA parameters.
///
/// Called automatically by [`rkf_init`] with `alpha = 0.05`, `initial_var = 1.0`.
#[unsafe(no_mangle)]
pub extern "C" fn rkf_init_tof_adaptive(alpha: f32, initial_var: f32) {
    unsafe {
        ptr::write(
            (*ADAPTIVE_TOF.0.get()).as_mut_ptr(),
            make_adaptive_tof(Tof::new(0.0, 0.0055), alpha, initial_var),
        );
    }
}

/// Fold an adaptive ToF measurement into the filter.
///
/// The measurement noise variance is estimated automatically by the EMA strategy;
/// no `stddev` argument is needed.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[unsafe(no_mangle)]
pub extern "C" fn rkf_update_tof_adaptive(distance: f32) {
    let ekf = unsafe { &mut *(*EKF.0.get()).as_mut_ptr() };
    let tof = unsafe { &mut *(*ADAPTIVE_TOF.0.get()).as_mut_ptr() };
    tof.tof_mut().distance = distance;
    ekf.correct(tof);
}

#[unsafe(no_mangle)]
pub extern "C" fn rkf_update_tof(distance: f32, stddev: f32) {
    let ekf = unsafe { &mut *(*EKF.0.get()).as_mut_ptr() };
    let mut tof = Tof::new(distance, stddev);
    ekf.correct(&mut tof);
    #[cfg(feature = "stats")]
    {
        let mut buf = [None::<f32>; 1];
        tof.last_residuals(&mut buf);
        unsafe { *TOF_RESIDUAL.0.get() = buf[0].unwrap_or(f32::NAN) };
        tof.last_innovation_covariance(&mut buf);
        unsafe { *TOF_INNOVATION_COVARIANCE.0.get() = buf[0].unwrap_or(f32::NAN) };
    }
}

/// Return the current estimated ToF noise (the value last used as R in the correction step).
///
/// Returns `NaN` before any adaptive correction has been applied.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[unsafe(no_mangle)]
pub extern "C" fn rkf_get_tof_adaptive_variance() -> f32 {
    let tof = unsafe { &*(*ADAPTIVE_TOF.0.get()).as_ptr() };
    tof.current_variances()[0].unwrap_or(f32::NAN)
}

/// Return the current estimated UWB noise for the given anchor slot (the value last used as R).
///
/// Returns `NaN` for out-of-range `anchor_id` or before the first adaptive correction.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[unsafe(no_mangle)]
pub extern "C" fn rkf_get_uwb_adaptive_variance(anchor_id: u8) -> f32 {
    if anchor_id as usize >= RKF_MAX_UWB_ANCHORS {
        return f32::NAN;
    }
    let anchors_ptr = unsafe { (*ADAPTIVE_ANCHORS.0.get()).as_ptr() as *const AdaptiveAnchor };
    let anchor = unsafe { &*anchors_ptr.add(anchor_id as usize) };
    anchor.current_variances()[0].unwrap_or(f32::NAN)
}

/// Write the 3×3 rotation matrix (body → NED) into `out` in row-major order.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
/// `out` must point to a `float[3][3]` array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rkf_get_rotation_matrix(out: *mut [f32; 3]) {
    let ekf = unsafe { &*(*EKF.0.get()).as_ptr() };
    let r = ekf.rotation_matrix();
    for i in 0..3 {
        for j in 0..3 {
            unsafe { (*out.add(i))[j] = r[(i, j)] };
        }
    }
}

/// Return the residual from the most recent `rkf_update_uwb` call.
///
/// Returns `NaN` before the first call.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[cfg(feature = "stats")]
#[unsafe(no_mangle)]
pub extern "C" fn rkf_get_uwb_residual() -> f32 {
    unsafe { *UWB_RESIDUAL.0.get() }
}

/// Return the residual from the most recent `rkf_update_tof` call.
///
/// Returns `NaN` if the measurement was rejected (tilt too steep) or before the
/// first call.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[cfg(feature = "stats")]
#[unsafe(no_mangle)]
pub extern "C" fn rkf_get_tof_residual() -> f32 {
    unsafe { *TOF_RESIDUAL.0.get() }
}

/// Return the residual from the most recent `rkf_update_uwb_adaptive` call for
/// the given anchor slot.
///
/// Returns `NaN` for out-of-range `anchor_id` or before the first call.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[cfg(feature = "stats")]
#[unsafe(no_mangle)]
pub extern "C" fn rkf_get_uwb_adaptive_residual(anchor_id: u8) -> f32 {
    if anchor_id as usize >= RKF_MAX_UWB_ANCHORS {
        return f32::NAN;
    }
    let anchors_ptr = unsafe { (*ADAPTIVE_ANCHORS.0.get()).as_ptr() as *const AdaptiveAnchor };
    let anchor = unsafe { &*anchors_ptr.add(anchor_id as usize) };
    let mut buf = [None::<f32>; 1];
    anchor.last_residuals(&mut buf);
    buf[0].unwrap_or(f32::NAN)
}

/// Return the residual from the most recent `rkf_update_tof_adaptive` call.
///
/// Returns `NaN` if the measurement was rejected (tilt too steep) or before the
/// first call.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[cfg(feature = "stats")]
#[unsafe(no_mangle)]
pub extern "C" fn rkf_get_tof_adaptive_residual() -> f32 {
    let tof = unsafe { &*(*ADAPTIVE_TOF.0.get()).as_ptr() };
    let mut buf = [None::<f32>; 1];
    tof.last_residuals(&mut buf);
    buf[0].unwrap_or(f32::NAN)
}

/// Return the innovation covariance (`s = H P Hᵀ + R`) from the most recent
/// `rkf_update_uwb` call.
///
/// Returns `NaN` before the first call.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[cfg(feature = "stats")]
#[unsafe(no_mangle)]
pub extern "C" fn rkf_get_uwb_innovation_covariance() -> f32 {
    unsafe { *UWB_INNOVATION_COVARIANCE.0.get() }
}

/// Return the innovation covariance (`s = H P Hᵀ + R`) from the most recent
/// `rkf_update_tof` call.
///
/// Returns `NaN` if the measurement was rejected (tilt too steep) or before the
/// first call.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[cfg(feature = "stats")]
#[unsafe(no_mangle)]
pub extern "C" fn rkf_get_tof_innovation_covariance() -> f32 {
    unsafe { *TOF_INNOVATION_COVARIANCE.0.get() }
}

/// Return the innovation covariance (`s = H P Hᵀ + R`) from the most recent
/// `rkf_update_uwb_adaptive` call for the given anchor slot.
///
/// Returns `NaN` for out-of-range `anchor_id` or before the first call.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[cfg(feature = "stats")]
#[unsafe(no_mangle)]
pub extern "C" fn rkf_get_uwb_adaptive_innovation_covariance(anchor_id: u8) -> f32 {
    if anchor_id as usize >= RKF_MAX_UWB_ANCHORS {
        return f32::NAN;
    }
    let anchors_ptr = unsafe { (*ADAPTIVE_ANCHORS.0.get()).as_ptr() as *const AdaptiveAnchor };
    let anchor = unsafe { &*anchors_ptr.add(anchor_id as usize) };
    let mut buf = [None::<f32>; 1];
    anchor.last_innovation_covariance(&mut buf);
    buf[0].unwrap_or(f32::NAN)
}

/// Return the innovation covariance (`s = H P Hᵀ + R`) from the most recent
/// `rkf_update_tof_adaptive` call.
///
/// Returns `NaN` if the measurement was rejected (tilt too steep) or before the
/// first call.
///
/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[cfg(feature = "stats")]
#[unsafe(no_mangle)]
pub extern "C" fn rkf_get_tof_adaptive_innovation_covariance() -> f32 {
    let tof = unsafe { &*(*ADAPTIVE_TOF.0.get()).as_ptr() };
    let mut buf = [None::<f32>; 1];
    tof.last_innovation_covariance(&mut buf);
    buf[0].unwrap_or(f32::NAN)
}

/// # Safety
///
/// [`rkf_init`] must have been called before this function.
#[unsafe(no_mangle)]
pub extern "C" fn rkf_get_state() -> RkfState {
    let ekf = unsafe { &*(*EKF.0.get()).as_ptr() };
    let q = ekf.attitude;
    let pos = ekf.position();
    let vel = ekf.velocity();

    // SAFETY: every field is written before `out` is returned.
    // Avoids the zero-init of `covariance` that [0.0; N] would emit.
    #[allow(invalid_value, clippy::uninit_assumed_init)]
    let mut out: RkfState = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
    out.position = [pos.x, pos.y, pos.z];
    out.velocity = [vel.x, vel.y, vel.z];
    out.attitude = [q.w, q.i, q.j, q.k];
    // nalgebra stores matrices column-major; write row-major into the C array.
    let cov_slice = ekf.covariance.as_slice();
    for i in 0..STATE_DIM {
        for j in 0..STATE_DIM {
            out.covariance[i * STATE_DIM + j] = cov_slice[j * STATE_DIM + i];
        }
    }
    out
}
