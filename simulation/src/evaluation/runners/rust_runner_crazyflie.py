from typing import Literal

import numpy as np
import scipy.constants
from ekf import (AccelConvention, AdaptiveTof, AdaptiveTofEma, AdaptiveUwbAnchor, AdaptiveUwbAnchorEma,
                 GeodeticOrigin, Imu, ImuNoise, Tof, UwbAnchor)
from scipy.spatial.transform import Rotation

from core import SimulationResult
from core.sensor_stream import SensorStream
from evaluation.core import KalmanRunner, KalmanEstimate
from evaluation.runners._ekf_helpers import (
    _R_ENU_TO_NED,
    _enu_to_ned_vec,
    _ned_to_enu_batch,
    _ned_to_geodetic,
    _find_sensor,
    _init_ekf_from_gt_crazyflie,
)


class RustRunnerCrazyflie(KalmanRunner):
    """EKF runner for Crazyflie simulation data.

    NOTE: the Rust EKF works in NED. Ground truth and KalmanEstimate
    use ENU. All conversions are done in this runner.
    """

    def __init__(
            self,
            accel_stddev: float = 0.3691,
            gyro_stddev: float = 0.0498,
            uwb_stddev: float = 0.0204,
            tof_stddev: float = 0.0068,
            use_noisy: bool = True,
            accel_sensor: str | None = None,
            gyro_sensor: str | None = None,
            uwb_sensors: list[str] | None = None,
            tof_sensor: str | None = None,
            geodetic_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
            do_correction_uwb: bool = True,
            do_correction_tof: bool = True,
            adaptive_uwb: Literal["sliding_window", "ema"] | None = None,
            uwb_ema_alpha: float = 0.05,
            uwb_ema_initial_var: float = 0.02,
            adaptive_tof: Literal["sliding_window", "ema"] | None = None,
            tof_ema_alpha: float = 0.05,
            tof_ema_initial_var: float = 0.02,
    ):
        """
        Args:
            accel_stddev:    Accelerometer process noise std dev [m/s²].
            gyro_stddev:     Gyroscope process noise std dev [rad/s].
            uwb_stddev:      UWB range measurement noise std dev [m].
            tof_stddev:      ToF height measurement noise std dev [m].
            use_noisy:       Use sensors_noisy (True) or sensors_clean (False).
            accel_sensor:    Explicit sensor key. Auto-detected if None.
            gyro_sensor:     Explicit sensor key. Auto-detected if None.
            uwb_sensors:     Explicit list of UWB sensor keys. Auto-detected if
                             None. UWB correction is skipped when none are found.
            tof_sensor:      Explicit ToF sensor key. Auto-detected if None.
                             ToF correction is skipped when no stream is found.
            geodetic_origin: (lat_deg, lon_deg, alt_m) used as the NED origin.
                             Defaults to (0, 0, 0) — arbitrary but consistent.
            do_correction_uwb:   Enable or disable UWB corrections.
            do_correction_tof:   Enable or disable ToF corrections.
            adaptive_uwb:    UWB noise adaptation mode.
                             None             – fixed noise (standard EKF).
                             "sliding_window" – innovation-based sliding-window R update.
                             "ema"            – exponential moving average R update.
            uwb_ema_alpha:           EMA smoothing factor = (0, 1]. Only used when adaptive_uwb="ema".
            uwb_ema_initial_var:     Initial variance for the UWB EMA estimator [m²]. Only used when adaptive_uwb="ema".
            adaptive_tof:        ToF noise adaptation mode.
                                 None             – fixed noise (standard EKF).
                                 "sliding_window" – innovation-based sliding-window R update.
                                 "ema"            – exponential moving average R update.
            tof_ema_alpha:       EMA smoothing factor ∈ (0, 1]. Only used when adaptive_tof="ema".
            tof_ema_initial_var: Initial variance for the ToF EMA estimator [m²]. Only used when adaptive_tof="ema".
        """
        if adaptive_uwb not in (None, "sliding_window", "ema"):
            raise ValueError(f"adaptive_uwb must be None, 'sliding_window', or 'ema', got {adaptive_uwb!r}")
        if adaptive_tof not in (None, "sliding_window", "ema"):
            raise ValueError(f"adaptive_tof must be None, 'sliding_window', or 'ema', got {adaptive_tof!r}")

        self._accel_stddev = accel_stddev
        self._gyro_stddev = gyro_stddev
        self._uwb_stddev = uwb_stddev
        self._tof_stddev = tof_stddev
        self._use_noisy = use_noisy
        self._accel_sensor = accel_sensor
        self._gyro_sensor = gyro_sensor
        self._uwb_sensors = uwb_sensors
        self._tof_sensor = tof_sensor
        self._geodetic_origin = geodetic_origin
        self._do_correction_uwb = do_correction_uwb
        self._do_correction_tof = do_correction_tof
        self._adaptive_uwb = adaptive_uwb
        self._uwb_ema_alpha = uwb_ema_alpha
        self._uwb_ema_initial_var = uwb_ema_initial_var
        self._adaptive_tof = adaptive_tof
        self._tof_ema_alpha = tof_ema_alpha
        self._tof_ema_initial_var = tof_ema_initial_var

    def run(self, simulation_result: SimulationResult) -> KalmanEstimate:
        gt = simulation_result.ground_truth
        sensors = simulation_result.sensors_noisy if self._use_noisy else simulation_result.sensors_clean

        accel_key = self._accel_sensor or _find_sensor(sensors, "accel")
        gyro_key = self._gyro_sensor or _find_sensor(sensors, "gyro")
        uwb_keys = self._uwb_sensors or _find_uwb_sensors(sensors)
        tof_key = self._tof_sensor or _find_sensor(sensors, "time_of_flight", required=False)

        accel_stream: SensorStream = sensors[accel_key]
        gyro_stream: SensorStream = sensors[gyro_key]
        uwb_streams: list[SensorStream] = [sensors[k] for k in uwb_keys]
        tof_stream: SensorStream | None = sensors[tof_key] if tof_key else None

        lat0, lon0, alt0 = self._geodetic_origin
        origin = GeodeticOrigin(lat_deg=lat0, lon_deg=lon0, alt_m=alt0)

        ekf = _init_ekf_from_gt_crazyflie(gt)
        imu_noise = ImuNoise(accel_stddev=self._accel_stddev, gyro_stddev=self._gyro_stddev)

        events = _build_event_queue(accel_stream, gyro_stream, uwb_streams, tof_stream)

        tof_measurement = self._make_tof_measurement()
        uwb_measurements: dict[int, object] = {}  # anchor_idx → persistent adaptive object

        times, pos_ned, vel_ned, att_ned, covs = [], [], [], [], []
        accel_body_log, gyro_body_log = [], []
        uwb_vars: list[np.ndarray] = []
        last_uwb_var = np.full(len(uwb_streams), np.nan)
        tof_vars: list[np.ndarray] = []
        last_tof_var = np.full(1, np.nan)
        log_tof = self._do_correction_tof and tof_key is not None

        last_time = float(gt.time[0])

        for event in events:
            dt = event["time"] - last_time
            if dt < 0.0:
                continue

            if event["type"] == "imu":
                last_time = event["time"]

                # Convert from Crazyflie units (G, deg/s) to SI (m/s², rad/s)
                accel_si = event["accel"] * scipy.constants.g
                gyro_si = np.deg2rad(event["gyro"])

                # Convert Body Frame: FLU (Crazyflie) -> FRD (like NED)
                accel_si[1] = -accel_si[1]
                accel_si[2] = -accel_si[2]
                gyro_si[1] = -gyro_si[1]
                gyro_si[2] = -gyro_si[2]

                if dt > 0.0:
                    predictor = Imu(
                        accel=accel_si,
                        gyro=gyro_si,
                        noise=imu_noise,
                        accel_convention=AccelConvention.SpecificForce,
                    )
                    ekf.predict(predictor, dt)

                times.append(event["time"])
                pos_ned.append(ekf.position.copy())
                vel_ned.append(ekf.velocity.copy())
                att_ned.append(ekf.attitude.copy())
                covs.append(ekf.covariance.copy())
                accel_body_log.append(accel_si.copy())
                gyro_body_log.append(gyro_si.copy())
                uwb_vars.append(last_uwb_var.copy())
                tof_vars.append(last_tof_var.copy())

            elif event["type"] == "uwb" and self._do_correction_uwb and not np.isnan(event["distance"]):
                anchor_enu = np.array([event["anchor_x"], event["anchor_y"], event["anchor_z"]])
                anchor_ned = _enu_to_ned_vec(anchor_enu)
                anchor_idx = event["anchor_idx"]
                if self._adaptive_uwb is not None:
                    if anchor_idx not in uwb_measurements:
                        uwb_measurements[anchor_idx] = self._make_uwb_measurement(anchor_ned, event["distance"])
                    else:
                        uwb_measurements[anchor_idx].distance = event["distance"]
                    measurement = uwb_measurements[anchor_idx]
                    ekf.correct(measurement)
                    last_uwb_var[anchor_idx] = measurement.current_variance
                else:
                    ekf.correct(UwbAnchor(ned=anchor_ned, distance=event["distance"], stddev=self._uwb_stddev))
                    last_uwb_var[anchor_idx] = self._uwb_stddev ** 2

            elif event["type"] == "tof" and self._do_correction_tof and not np.isnan(event["distance"]):
                if tof_measurement is not None:
                    tof_measurement.distance = event["distance"]
                    ekf.correct(tof_measurement)
                    last_tof_var[0] = tof_measurement.current_variance
                else:
                    ekf.correct(Tof(distance=event["distance"], stddev=self._tof_stddev))
                    last_tof_var[0] = self._tof_stddev ** 2

        ekf.finalize()

        pos_ned_arr = np.array(pos_ned)
        vel_ned_arr = np.array(vel_ned)
        att_ned_arr = np.array(att_ned)
        accel_body_arr = np.array(accel_body_log)
        gyro_body_arr = np.array(gyro_body_log)

        pos_enu = _ned_to_enu_batch(pos_ned_arr)
        vel_enu = _ned_to_enu_batch(vel_ned_arr)

        _R_FLU_TO_FRD = Rotation.from_matrix(np.diag([1.0, -1.0, -1.0]))
        r_body_ned = Rotation.from_quat(att_ned_arr, scalar_first=True)

        r_body_enu = _R_ENU_TO_NED.inv() * r_body_ned * _R_FLU_TO_FRD
        att_enu = r_body_enu.as_quat(scalar_first=True)

        accel_enu = r_body_enu.apply(accel_body_arr)
        geo_coords = _ned_to_geodetic(pos_ned_arr, origin, alt0)

        mvar: dict[str, np.ndarray] = {"uwb": np.array(uwb_vars)}
        if log_tof:
            mvar["tof"] = np.array(tof_vars)

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
            measurement_variances=mvar,
            metadata={
                "runner": "RustRunnerCrazyflie",
                "runner_config": self.describe(),
                "accel_sensor": accel_key,
                "gyro_sensor": gyro_key,
                "uwb_sensors": uwb_keys,
                "tof_sensor": tof_key,
                "use_noisy": self._use_noisy,
            },
        )

    def describe(self) -> str:
        uwb_adaptive = self._adaptive_uwb or "none"
        uwb_ema_info = f", ema_alpha={self._uwb_ema_alpha}, ema_initial_var={self._uwb_ema_initial_var}" \
            if self._adaptive_uwb == "ema" else ""
        tof_adaptive = self._adaptive_tof or "none"
        tof_ema_info = f", tof_ema_alpha={self._tof_ema_alpha}, tof_ema_initial_var={self._tof_ema_initial_var}" \
            if self._adaptive_tof == "ema" else ""
        return (
            f"RustRunnerCrazyflie("
            f"accel_stddev={self._accel_stddev}, "
            f"gyro_stddev={self._gyro_stddev}, "
            f"uwb_stddev={self._uwb_stddev}, "
            f"tof_stddev={self._tof_stddev}, "
            f"use_noisy={self._use_noisy}, "
            f"do_correction_uwb={self._do_correction_uwb}, "
            f"do_correction_tof={self._do_correction_tof}, "
            f"adaptive_uwb={uwb_adaptive}{uwb_ema_info}, "
            f"adaptive_tof={tof_adaptive}{tof_ema_info}, "
            f"geodetic_origin={self._geodetic_origin})"
        )

    def _make_tof_measurement(self) -> AdaptiveTof | AdaptiveTofEma | None:
        if self._adaptive_tof == "sliding_window":
            return AdaptiveTof(distance=0.0, stddev=self._tof_stddev)
        elif self._adaptive_tof == "ema":
            return AdaptiveTofEma(
                distance=0.0, stddev=self._tof_stddev,
                alpha=self._tof_ema_alpha, initial_var=self._tof_ema_initial_var,
            )
        return None

    def _make_uwb_measurement(self, anchor_ned: np.ndarray, distance: float):
        if self._adaptive_uwb == "sliding_window":
            return AdaptiveUwbAnchor(ned=anchor_ned, distance=distance, stddev=self._uwb_stddev)
        elif self._adaptive_uwb == "ema":
            return AdaptiveUwbAnchorEma(
                ned=anchor_ned, distance=distance, stddev=self._uwb_stddev,
                alpha=self._uwb_ema_alpha, initial_var=self._uwb_ema_initial_var,
            )
        return UwbAnchor(ned=anchor_ned, distance=distance, stddev=self._uwb_stddev)


# ------------------------------------------------------------------
# Runner-specific helpers
# ------------------------------------------------------------------

def _find_uwb_sensors(sensors: dict) -> list[str]:
    """Return all sensor keys that contain 'uwb' (case-insensitive)."""
    return [k for k in sensors if "uwb" in k.lower()]


def _build_event_queue(
        accel_stream: SensorStream,
        gyro_stream: SensorStream,
        uwb_streams: list[SensorStream],
        tof_stream: SensorStream | None = None,
) -> list[dict]:
    """Merge IMU, UWB, and ToF events into a single time-sorted list."""
    events: list[dict] = []

    for i in range(len(accel_stream.time)):
        events.append({
            "time": float(accel_stream.time[i]),
            "type": "imu",
            "accel": accel_stream.data[i].astype(np.float64),
            "gyro": gyro_stream.data[i].astype(np.float64),
        })

    for anchor_idx, stream in enumerate(uwb_streams):
        for i in range(len(stream.time)):
            events.append({
                "time": float(stream.time[i]),
                "type": "uwb",
                "anchor_idx": anchor_idx,
                "distance": float(stream.data[i, 0]),
                "anchor_x": float(stream.data[i, 2]),
                "anchor_y": float(stream.data[i, 3]),
                "anchor_z": float(stream.data[i, 4]),
            })

    if tof_stream is not None:
        for i in range(len(tof_stream.time)):
            events.append({
                "time": float(tof_stream.time[i]),
                "type": "tof",
                "distance": float(tof_stream.data[i, 0]),
            })

    events.sort(key=lambda e: e["time"])
    return events
