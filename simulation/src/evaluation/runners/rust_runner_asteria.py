import math
import warnings
from typing import Literal

import numpy as np
from ekf import (
    AccelConvention, AdaptiveGnss, AdaptiveGnssEma, GeodeticOrigin, Gnss, Imu, ImuNoise,
    IDX_N, IDX_E, IDX_D, IDX_VN, IDX_VE, IDX_VD, IDX_AX, IDX_AY, IDX_AZ,
)
from scipy.spatial.transform import Rotation

from core import SimulationResult
from core.sensor_stream import SensorStream
from evaluation.core import KalmanRunner, KalmanEstimate
from evaluation.runners._ekf_helpers import (
    _R_ENU_TO_NED,
    _ned_to_enu_batch,
    _ned_to_geodetic,
    _find_sensor,
    _init_ekf_from_gt_asteria, )


class RustRunnerAsteria(KalmanRunner):
    """EKF runner V1 (usable for rocket data).

    NOTE: the Rust EKF works in NED; ground truth and KalmanEstimate
    use ENU. Conversions are all done in this runner.
    """

    def __init__(
            self,
            accel_stddev: float = 0.29496,
            gyro_stddev: float = 0.0285,
            gnss_stddev: np.ndarray = np.array([2.312, 2.312, 3.8472]),
            use_noisy: bool = True,
            accel_sensor: str | None = None,
            gyro_sensor: str | None = None,
            gnss_sensor: str | None = None,
            do_correction: bool = True,
            adaptive_gnss: Literal["sliding_window", "ema"] | None = None,
            ema_alpha: float = 0.05,
            ema_initial_var: float = 4.0,
    ):
        """
        Args:
            accel_stddev:    Accelerometer process noise std dev [m/s²].
            gyro_stddev:     Gyroscope process noise std dev [rad/s].
            gnss_stddev:     GNSS measurement noise std dev per NED axis [m].
            use_noisy:       Use sensors_noisy (True) or sensors_clean (False).
            accel_sensor:    Explicit sensor key. Auto-detected if None.
            gyro_sensor:     Explicit sensor key. Auto-detected if None.
            gnss_sensor:     Explicit sensor key. Auto-detected if None.
                             GNSS correction is skipped when no key is found.
            do_correction:   Enable or disable GNSS correction.
            adaptive_gnss:   GNSS noise adaptation mode.
                             None             – fixed noise (standard EKF).
                             "sliding_window" – innovation-based sliding-window R update.
                             "ema"            – exponential moving average R update.
            ema_alpha:       EMA smoothing factor = (0, 1]. Only used when adaptive_gnss="ema".
                             Higher values weight recent innovations more.
            ema_initial_var: Initial variance for the EMA estimator [m²]. Only used when adaptive_gnss="ema".
        """
        if adaptive_gnss not in (None, "sliding_window", "ema"):
            raise ValueError(f"adaptive_gnss must be None, 'sliding_window', or 'ema', got {adaptive_gnss!r}")

        self._accel_stddev = accel_stddev
        self._gyro_stddev = gyro_stddev
        self._gnss_stddev = gnss_stddev
        self._use_noisy = use_noisy
        self._accel_sensor = accel_sensor
        self._gyro_sensor = gyro_sensor
        self._gnss_sensor = gnss_sensor
        self._do_correction = do_correction
        self._adaptive_gnss = adaptive_gnss
        self._ema_alpha = ema_alpha
        self._ema_initial_var = ema_initial_var

    def run(self, simulation_result: SimulationResult) -> KalmanEstimate:
        gt = simulation_result.ground_truth
        sensors = simulation_result.sensors_noisy if self._use_noisy else simulation_result.sensors_clean

        accel_key = self._accel_sensor or _find_sensor(sensors, "accelerometer")
        gyro_key = self._gyro_sensor or _find_sensor(sensors, "gyroscope")
        gnss_key = self._gnss_sensor or _find_sensor(sensors, "gnss", required=False)

        accel_stream: SensorStream = sensors[accel_key]
        gyro_stream: SensorStream = sensors[gyro_key]
        gnss_stream: SensorStream | None = sensors.get(gnss_key) if gnss_key else None

        lat0, lon0, alt0 = gt.geo_coordinates[0].astype(float)

        if math.isclose(lat0, 0) and math.isclose(lon0, 0):
            warnings.warn("Origin position seems off — first ground truth sample may be (0, 0, 0).")

        origin = GeodeticOrigin(lat_deg=lat0, lon_deg=lon0, alt_m=alt0)

        ekf = _init_ekf_from_gt_asteria(gt)
        zero_gyro_after = simulation_result.metadata.get("drogue_inflation_time")
        events = _build_event_queue(accel_stream, gyro_stream, gnss_stream, zero_gyro_after=zero_gyro_after)

        imu_noise = ImuNoise(accel_stddev=self._accel_stddev, gyro_stddev=self._gyro_stddev)
        gnss_measurement = self._make_gnss_measurement()

        times, pos_ned, vel_ned, att_ned, covs = [], [], [], [], []
        accel_body_log, gyro_body_log = [], []
        gnss_vars: list[np.ndarray] = []
        last_gnss_var = np.full(3, np.nan)

        last_time = float(gt.time[0])
        attitude_frozen = False

        q_gt_enu_at_freeze: np.ndarray | None = None
        if zero_gyro_after is not None:
            freeze_idx = max(np.searchsorted(gt.time, zero_gyro_after) - 1, 0)
            q_gt_enu_at_freeze = gt.attitude[freeze_idx]

        for event in events:
            dt = event["time"] - last_time
            if dt < 0.0:
                continue
            last_time = event["time"]

            if event["type"] == "imu":
                if dt > 0.0:
                    predictor = Imu(
                        accel=event["accel"],
                        gyro=event["gyro"],
                        noise=imu_noise,
                        accel_convention=AccelConvention.Coordinate,
                    )
                    ekf.predict(predictor, dt)
                    if attitude_frozen:
                        _clamp_attitude_covariance(ekf)

                if not attitude_frozen and zero_gyro_after is not None and event["time"] > zero_gyro_after:
                    _freeze_attitude(ekf, q_gt_enu_at_freeze)
                    attitude_frozen = True

                times.append(event["time"])
                pos_ned.append(ekf.position.copy())
                vel_ned.append(ekf.velocity.copy())
                att_ned.append(ekf.attitude.copy())
                covs.append(ekf.covariance.copy())
                gnss_vars.append(last_gnss_var.copy())
                accel_body_log.append(event["accel"].copy())
                gyro_body_log.append(event["gyro"].copy())

            elif event["type"] == "gnss" and self._do_correction and gnss_stream is not None and not any(
                    np.isnan(event[k]) for k in ("lat", "lon", "alt")):
                if gnss_measurement is not None:
                    gnss_measurement.set_geodetic(event["lat"], event["lon"], event["alt"], origin)
                    ekf.correct(gnss_measurement)
                    last_gnss_var = np.asarray(gnss_measurement.current_variances, dtype=np.float64)
                else:
                    ekf.correct(
                        Gnss.from_geodetic(event["lat"], event["lon"], event["alt"], origin, self._gnss_stddev))
                    last_gnss_var = self._gnss_stddev ** 2

        ekf.finalize()

        pos_ned_arr = np.array(pos_ned)
        vel_ned_arr = np.array(vel_ned)
        att_ned_arr = np.array(att_ned)
        accel_body_arr = np.array(accel_body_log)
        gyro_body_arr = np.array(gyro_body_log)

        pos_enu = _ned_to_enu_batch(pos_ned_arr)
        vel_enu = _ned_to_enu_batch(vel_ned_arr)

        r_body_ned = Rotation.from_quat(att_ned_arr, scalar_first=True)
        r_body_enu = _R_ENU_TO_NED.inv() * r_body_ned
        att_enu = r_body_enu.as_quat(scalar_first=True)

        accel_enu = r_body_enu.apply(accel_body_arr)
        geo_coords = _ned_to_geodetic(pos_ned_arr, origin, alt0)

        N = len(times)
        return KalmanEstimate(
            time=np.array(times),
            position=pos_enu,
            geo_coordinates=geo_coords,
            velocity=vel_enu,
            acceleration=accel_enu,
            attitude=att_enu,
            angular_velocity=gyro_body_arr,
            pressure=np.full(N, np.nan),
            estimation_output_data_rate=float(accel_stream.sampling_rate),
            covariance=np.array(covs),
            measurement_variances={"gnss": np.array(gnss_vars)},
            metadata={
                "runner": "RustRunnerV1",
                "runner_config": self.describe(),
                "accel_sensor": accel_key,
                "gyro_sensor": gyro_key,
                "gnss_sensor": gnss_key,
                "use_noisy": self._use_noisy,
            },
        )

    def describe(self) -> str:
        adaptive = self._adaptive_gnss or "none"
        ema_info = f", ema_alpha={self._ema_alpha}, ema_initial_var={self._ema_initial_var}" \
            if self._adaptive_gnss == "ema" else ""
        return (
            f"RustRunnerV1("
            f"accel_stddev={self._accel_stddev}, "
            f"gyro_stddev={self._gyro_stddev}, "
            f"gnss_stddev={self._gnss_stddev.tolist()}, "
            f"use_noisy={self._use_noisy}, "
            f"do_correction={self._do_correction}, "
            f"adaptive_gnss={adaptive}{ema_info})"
        )

    def _make_gnss_measurement(self):
        if self._adaptive_gnss == "sliding_window":
            return AdaptiveGnss(ned=np.zeros(3), stddev=self._gnss_stddev)
        elif self._adaptive_gnss == "ema":
            return AdaptiveGnssEma(
                ned=np.zeros(3), stddev=self._gnss_stddev,
                alpha=self._ema_alpha, initial_var=self._ema_initial_var,
            )
        return None


# ------------------------------------------------------------------
# Runner-specific helpers
# ------------------------------------------------------------------

_ATT_IDXS = np.array([IDX_AX, IDX_AY, IDX_AZ])
_NAV_IDXS = np.array([IDX_N, IDX_E, IDX_D, IDX_VN, IDX_VE, IDX_VD])


def _freeze_attitude(ekf, q_gt_enu: np.ndarray | None) -> None:
    """One-time: snap nominal attitude to GT then clamp covariance."""
    if q_gt_enu is not None:
        r_body_ned = _R_ENU_TO_NED * Rotation.from_quat(q_gt_enu, scalar_first=True)
        ekf.attitude = r_body_ned.as_quat(scalar_first=True)
    _clamp_attitude_covariance(ekf)


def _clamp_attitude_covariance(ekf) -> None:
    """Zero attitude cross-covariance. Called after every predict while frozen."""
    cov = ekf.covariance.copy()
    cov[np.ix_(_ATT_IDXS, _NAV_IDXS)] = 0.0
    cov[np.ix_(_NAV_IDXS, _ATT_IDXS)] = 0.0
    cov[np.ix_(_ATT_IDXS, _ATT_IDXS)] = np.eye(3) * 1e-6
    ekf.covariance = cov


def _build_event_queue(
        accel_stream: SensorStream,
        gyro_stream: SensorStream,
        gnss_stream: SensorStream | None,
        zero_gyro_after: float | None = None,
) -> list[dict]:
    """Merge IMU and GNSS events into a single time-sorted list."""
    events: list[dict] = []

    for i in range(len(accel_stream.time)):
        t = float(accel_stream.time[i])
        gyro = np.zeros(3, dtype=np.float64) if (zero_gyro_after is not None and t > zero_gyro_after) else \
            gyro_stream.data[i].astype(np.float64)
        events.append({
            "time": t,
            "type": "imu",
            "accel": accel_stream.data[i].astype(np.float64),
            "gyro": gyro,
        })

    if gnss_stream is not None:
        for i in range(len(gnss_stream.time)):
            events.append({
                "time": float(gnss_stream.time[i]),
                "type": "gnss",
                "lat": float(gnss_stream.data[i, 0]),
                "lon": float(gnss_stream.data[i, 1]),
                "alt": float(gnss_stream.data[i, 2]),
            })

    events.sort(key=lambda e: e["time"])
    return events
