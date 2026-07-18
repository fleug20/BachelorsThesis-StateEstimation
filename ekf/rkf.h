#ifndef RKF_H
#define RKF_H

#include <stdint.h>

#define STATE_DIM 9

#define IDX_N 0

#define IDX_E 1

#define IDX_D 2

#define IDX_VN 3

#define IDX_VE 4

#define IDX_VD 5

#define IDX_AX 6

#define IDX_AY 7

#define IDX_AZ 8

#define COVARIANCE_DIM (STATE_DIM * STATE_DIM)

/**
 * Initial diagonal variance for position states.
 */
#define MAX_COVARIANCE 200.0

/**
 * Minimum allowed diagonal variance (prevents singular covariance).
 */
#define MIN_COVARIANCE 1e-6

/**
 * Maximum number of UWB anchors tracked by the adaptive filter.
 */
#define RKF_MAX_UWB_ANCHORS 8

typedef struct RkfState {
  float position[3];
  float velocity[3];
  float attitude[4];
  float covariance[COVARIANCE_DIM];
} RkfState;

/**
 * Initialise the filter and the IMU scratch space.
 *
 * Must be called exactly once before any other `rkf_*` function.
 *
 * # Safety
 *
 * Not thread-safe. Call from a single context before starting any tasks that
 * use the other `rkf_*` functions.
 */
void rkf_init(void);

/**
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
void rkf_predict_imu(float ax,
                     float ay,
                     float az,
                     float gx,
                     float gy,
                     float gz,
                     float accel_std,
                     float gyro_std,
                     float dt);

void rkf_update_uwb(float anchor_n, float anchor_e, float anchor_d, float distance, float stddev);

/**
 * (Re-)initialise all adaptive UWB anchor states with the given EMA parameters.
 *
 * Called automatically by [`rkf_init`] with `alpha = 0.1`, `initial_var = 1.0`.
 * Call again to override those defaults before the first [`rkf_update_uwb_adaptive`].
 */
void rkf_init_uwb_adaptive(float alpha, float initial_var);

/**
 * Fold an adaptive UWB measurement into the filter.
 *
 * `anchor_id` selects which adaptive state slot to use (0 …
 * `RKF_MAX_UWB_ANCHORS - 1`). The call is a no-op for out-of-range IDs.
 * The measurement noise variance is estimated automatically by the EMA
 * strategy; no `stddev` argument is needed.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
void rkf_update_uwb_adaptive(uint8_t anchor_id,
                             float anchor_n,
                             float anchor_e,
                             float anchor_d,
                             float distance);

/**
 * (Re-)initialise the adaptive ToF state with the given EMA parameters.
 *
 * Called automatically by [`rkf_init`] with `alpha = 0.05`, `initial_var = 1.0`.
 */
void rkf_init_tof_adaptive(float alpha, float initial_var);

/**
 * Fold an adaptive ToF measurement into the filter.
 *
 * The measurement noise variance is estimated automatically by the EMA strategy;
 * no `stddev` argument is needed.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
void rkf_update_tof_adaptive(float distance);

void rkf_update_tof(float distance, float stddev);

/**
 * Return the current estimated ToF noise (the value last used as R in the correction step).
 *
 * Returns `NaN` before any adaptive correction has been applied.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
float rkf_get_tof_adaptive_variance(void);

/**
 * Return the current estimated UWB noise for the given anchor slot (the value last used as R).
 *
 * Returns `NaN` for out-of-range `anchor_id` or before the first adaptive correction.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
float rkf_get_uwb_adaptive_variance(uint8_t anchor_id);

/**
 * Write the 3×3 rotation matrix (body → NED) into `out` in row-major order.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 * `out` must point to a `float[3][3]` array.
 */
void rkf_get_rotation_matrix(float (*out)[3]);

/**
 * Return the residual from the most recent `rkf_update_uwb` call.
 *
 * Returns `NaN` before the first call.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
float rkf_get_uwb_residual(void);

/**
 * Return the residual from the most recent `rkf_update_tof` call.
 *
 * Returns `NaN` if the measurement was rejected (tilt too steep) or before the
 * first call.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
float rkf_get_tof_residual(void);

/**
 * Return the residual from the most recent `rkf_update_uwb_adaptive` call for
 * the given anchor slot.
 *
 * Returns `NaN` for out-of-range `anchor_id` or before the first call.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
float rkf_get_uwb_adaptive_residual(uint8_t anchor_id);

/**
 * Return the residual from the most recent `rkf_update_tof_adaptive` call.
 *
 * Returns `NaN` if the measurement was rejected (tilt too steep) or before the
 * first call.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
float rkf_get_tof_adaptive_residual(void);

/**
 * Return the innovation covariance (`s = H P Hᵀ + R`) from the most recent
 * `rkf_update_uwb` call.
 *
 * Returns `NaN` before the first call.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
float rkf_get_uwb_innovation_covariance(void);

/**
 * Return the innovation covariance (`s = H P Hᵀ + R`) from the most recent
 * `rkf_update_tof` call.
 *
 * Returns `NaN` if the measurement was rejected (tilt too steep) or before the
 * first call.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
float rkf_get_tof_innovation_covariance(void);

/**
 * Return the innovation covariance (`s = H P Hᵀ + R`) from the most recent
 * `rkf_update_uwb_adaptive` call for the given anchor slot.
 *
 * Returns `NaN` for out-of-range `anchor_id` or before the first call.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
float rkf_get_uwb_adaptive_innovation_covariance(uint8_t anchor_id);

/**
 * Return the innovation covariance (`s = H P Hᵀ + R`) from the most recent
 * `rkf_update_tof_adaptive` call.
 *
 * Returns `NaN` if the measurement was rejected (tilt too steep) or before the
 * first call.
 *
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
float rkf_get_tof_adaptive_innovation_covariance(void);

/**
 * # Safety
 *
 * [`rkf_init`] must have been called before this function.
 */
struct RkfState rkf_get_state(void);

#endif  /* RKF_H */
