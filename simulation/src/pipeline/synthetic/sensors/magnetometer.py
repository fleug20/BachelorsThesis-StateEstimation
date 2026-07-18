import datetime
import math
import warnings

import numpy as np
from pygeomag import GeoMag
from scipy.spatial.transform import Rotation

from core import GroundTruth, SensorStream, SensorOrigin
from pipeline.synthetic import SyntheticSensor


class SyntheticMagnetometer(SyntheticSensor):
    def __init__(
            self,
            name: str = "magnetometer",
            sampling_rate: float = 100.0,
            launch_site_elevation_m: float = None,
            launch_year: float = None,
    ):

        self.name = name
        self.sampling_rate = sampling_rate
        self.launch_site_elevation_m: float = launch_site_elevation_m
        self.launch_year: float = launch_year
        self.geo_mag = GeoMag()

        if self.launch_site_elevation_m is None:
            raise ValueError("Launch site elevation not set... please provide a value")
        if self.launch_year is None:
            self.launch_year = SyntheticMagnetometer._to_decimal_year(datetime.datetime.now())
            warnings.warn(f"Launch year not set... using: {self.launch_year}")

    def generate(self, ground_truth: GroundTruth) -> SensorStream:

        self._check_sampling_rate(ground_truth.simulation_sampling_rate)
        step: int = int(ground_truth.simulation_sampling_rate / self.sampling_rate)

        time = ground_truth.time[::step].copy()
        channels = ["bx", "by", "bz"]

        lat = ground_truth.geo_coordinates[::step, 0].copy()
        lon = ground_truth.geo_coordinates[::step, 1].copy()
        km_above_sea_level = (ground_truth.geo_coordinates[::step, 2].copy() + self.launch_site_elevation_m) / 1000

        data_ned = self.earth_field_ned(lat, lon, km_above_sea_level, self.launch_year)
        # Convert NED -> ENU to match RocketPy's inertial frame
        data_enu = np.empty_like(data_ned)
        data_enu[:, 0] = data_ned[:, 1]  # East  =  NED.y
        data_enu[:, 1] = data_ned[:, 0]  # North =  NED.x
        data_enu[:, 2] = -data_ned[:, 2]  # Up    = -NED.z

        # RocketPy quaternion is body -> ENU, so inverse is ENU -> body
        attitudes = Rotation.from_quat(ground_truth.attitude[::step], scalar_first=True)
        data_body = attitudes.inv().apply(data_enu)

        data = data_body / 100  # nT -> mgauss

        unit = "mgauss"
        origin = SensorOrigin.SYNTHETIC
        processing_history = [
            f"Sensor data by SyntheticSensorGenerator. Sensor origin: {origin}, Sampling rate: {str(self.sampling_rate)} Hz, Data units: {unit}"
        ]

        return SensorStream(name=self.name,
                            time=time,
                            data=data,
                            channels=channels,
                            unit=unit,
                            sampling_rate=self.sampling_rate,
                            origin=origin,
                            processing_history=processing_history)

    def _check_sampling_rate(self, simulation_sampling_rate: float):
        ratio = simulation_sampling_rate / self.sampling_rate
        if not math.isclose(ratio, round(ratio)):
            raise ValueError(
                f"Interpolation of Ground Truth not implemented yet... Ground truth sampling rate must be a multiple of the sensor sampling rate ({self.sampling_rate})"
            )

    def earth_field_ned(self, lat_deg: np.ndarray, lon_deg: np.ndarray, alt_km: np.ndarray, year: float):
        lat = np.atleast_1d(lat_deg).ravel()
        lon = np.atleast_1d(lon_deg).ravel()
        alt = np.atleast_1d(alt_km).ravel()
        lat, lon, alt = np.broadcast_arrays(lat, lon, alt)

        out = np.empty((lat.size, 3))
        for i in range(lat.size):
            r = self.geo_mag.calculate(float(lat[i]), float(lon[i]), float(alt[i]), year)
            out[i] = (r.x, r.y, r.z)

        return out if out.shape[0] > 1 else out[0]

    @classmethod
    def _to_decimal_year(cls, dt: datetime.datetime) -> float:
        start = datetime.datetime(dt.year, 1, 1)
        end = datetime.datetime(dt.year + 1, 1, 1)
        return dt.year + (dt - start).total_seconds() / (end - start).total_seconds()
