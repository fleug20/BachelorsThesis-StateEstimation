import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pymap3d as pm
from rocketpy import Flight, Environment, Rocket

from core import GroundTruth, SensorStream, SensorOrigin, SimulationResult
from rocketpy_simulation.base import RocketBase, EnvironmentBase

SENSOR_CHANNELS = {
    "Accelerometer": ["ax", "ay", "az"],  # m/s^2
    "Gyroscope": ["wx", "wy", "wz"],  # rad/s
    "Barometer": ["pressure"],  # Pa
    "GnssReceiver": ["latitude", "longitude", "altitude"],
}


@dataclass
class RocketSimulation:
    """Main class for running a rocket simulation with RocketPy."""

    environment_builder: EnvironmentBase = None
    rocket_builder: RocketBase = None

    # Flight parameters
    rail_length: float = 15.0  # m
    inclination: float = 85.0  # degrees
    heading: float = 144.0  # degrees

    # Simulation Parameters
    simulation_sampling_rate: float = 800.0  # Hz

    # Simulation
    flight: Flight = field(default=None, init=False)

    def run(self, name: str) -> SimulationResult:
        print("RocketSimulation: running simulation...")
        self._simulate()

        ground_truth: GroundTruth = self._get_ground_truth()
        clean, noisy = self._get_sensor_streams()
        has_parachutes = len(self.flight.parachute_events) >= 2
        metadata = {
            "simulator": "RocketSimulation",
            "name": name,
            "simulation_datetime": datetime.now(),
            "simulation_sampling_rate": self.simulation_sampling_rate,
            "out_of_rail_time": float(self.flight.out_of_rail_time),
            "apogee_time": float(self.flight.apogee_time),
            "drogue_deployment_time": float(self.flight.parachute_events[0][0]) if has_parachutes else None,
            "drogue_inflation_time": float(self.flight.parachute_events[0][0] + 1.5) if has_parachutes else None,
            "main_parachute_deployment_time": float(self.flight.parachute_events[1][0]) if has_parachutes else None,
            "main_parachute_inflation_time": float(self.flight.parachute_events[1][0] + 1.5) if has_parachutes else None,
        }

        return SimulationResult(ground_truth=ground_truth, sensors_clean=clean, sensors_noisy=noisy, metadata=metadata)

    def _simulate(self) -> None:
        env: Environment = self.environment_builder.build()
        rocket: Rocket = self.rocket_builder.build()

        self.flight = Flight(
            rocket=rocket,
            environment=env,
            rail_length=self.rail_length,
            inclination=self.inclination,
            heading=self.heading,
            time_overshoot=False,
            max_time=700,
            rtol=1e-10,
            atol=1e-10,
            ode_solver="RK45",
        )

    def _get_ground_truth(self) -> GroundTruth:
        flight = self.flight
        time = np.arange(0, flight.time[-1], 1 / self.simulation_sampling_rate)

        # NOTE: get_source() returns shape (N, 2) where col 0 = time, col 1 = value -> thus only use second values
        position = np.column_stack([
            flight.x(time),
            flight.y(time),
            # use altitude above ground instead of z as it starts at 0
            flight.altitude(time),
        ])
        # Use WGS84 ellipsoid (pymap3d) instead of RocketPy's Haversine (spherical) approximation
        x = flight.x(time)
        y = flight.y(time)
        alt_above_ground = flight.altitude(time)
        lat0 = self.environment_builder.latitude
        lon0 = self.environment_builder.longitude
        h0 = self.environment_builder.elevation
        wgs84_lat, wgs84_lon, wgs84_alt = pm.enu2geodetic(x, y, alt_above_ground, lat0, lon0, h0)
        geo_coordinates = np.column_stack([wgs84_lat, wgs84_lon, wgs84_alt])

        velocity = np.column_stack([
            flight.vx(time),
            flight.vy(time),
            flight.vz(time),
        ])
        acceleration = np.column_stack([
            flight.ax(time),
            flight.ay(time),
            flight.az(time),
        ])
        attitude = np.column_stack([
            flight.e0(time),
            flight.e1(time),
            flight.e2(time),
            flight.e3(time),
        ])
        if flight.parachute_events:
            drogue_inflation_time = float(flight.parachute_events[0][0]) + 1.5
            attitude = self._freeze_attitude_after(attitude, time, drogue_inflation_time)
        angular_velocity = np.column_stack([
            flight.w1(time),
            flight.w2(time),
            flight.w3(time),
        ])
        pressure = flight.pressure(time)

        return GroundTruth(
            time=time,
            position=position,
            geo_coordinates=geo_coordinates,
            velocity=velocity,
            acceleration=acceleration,
            attitude=self._normalize_quaternion_sign(attitude),
            angular_velocity=angular_velocity,
            pressure=pressure,
            simulation_sampling_rate=self.simulation_sampling_rate
        )

    def _get_sensor_streams(self) -> tuple[dict[str, SensorStream], dict[str, SensorStream]]:
        clean: dict[str, SensorStream] = {}
        noisy: dict[str, SensorStream] = {}

        for sensor in self.flight.sensors:
            type_name = type(sensor).__name__

            if type_name not in SENSOR_CHANNELS:
                raise ValueError(f"Sensor type '{type_name}' currently not supported.")

            name = sensor.name
            is_clean = name.endswith("_clean")
            origin = SensorOrigin.SIMULATED_CLEAN if is_clean else SensorOrigin.SIMULATED_NOISY

            time = np.array(sensor.measured_data)[:, 0]
            data = np.array(sensor.measured_data)[:, 1:]

            # avoid having time duplicates
            accumulate_max = np.maximum.accumulate(time)
            valid_mask = np.concatenate(([True], time[1:] > accumulate_max[:-1]))
            if not np.all(valid_mask):
                time = time[valid_mask]
                data = data[valid_mask]
                warnings.warn(
                    f"Warning: Sensor '{name}' has duplicate time values generated by RocketPy."
                )

            # Use WGS84 ellipsoid (pymap3d) instead of RocketPy's Haversine (spherical) approximation
            if type_name == "GnssReceiver":
                data = self._gnss_haversine_to_wgs84(time, data)
            channels = SENSOR_CHANNELS[type_name]
            unit = sensor.units
            sampling_rate = sensor.sampling_rate
            processing_history = [
                f"Sensor data by RocketPy. Original name: {name}, Sensor origin: {origin}, Sampling rate: {str(sampling_rate)} Hz, Data units: {unit}"
            ]

            stream = SensorStream(name=name,
                                  time=time,
                                  data=data,
                                  channels=channels,
                                  unit=unit,
                                  sampling_rate=sampling_rate,
                                  origin=origin,
                                  processing_history=processing_history, )

            if is_clean:
                clean[name] = stream
            else:
                noisy[name] = stream

        return clean, noisy

    def _gnss_haversine_to_wgs84(self, sensor_times: np.ndarray, data: np.ndarray) -> np.ndarray:
        lat0 = self.environment_builder.latitude
        lon0 = self.environment_builder.longitude
        h0 = self.environment_builder.elevation

        x = self.flight.x(sensor_times)
        y = self.flight.y(sensor_times)
        alt_above_ground = self.flight.altitude(sensor_times)

        # True position in both systems at each sample time
        true_wgs84_lat, true_wgs84_lon, true_wgs84_alt = pm.enu2geodetic(x, y, alt_above_ground, lat0, lon0, h0)
        true_hav_lat = self.flight.latitude(sensor_times)
        true_hav_lon = self.flight.longitude(sensor_times)
        true_hav_alt = alt_above_ground + h0

        # Isolate noise added by the sensor, then re-apply it on top of the WGS84 true position
        result = data.copy()
        result[:, 0] = true_wgs84_lat + (data[:, 0] - true_hav_lat)
        result[:, 1] = true_wgs84_lon + (data[:, 1] - true_hav_lon)
        result[:, 2] = true_wgs84_alt + (data[:, 2] - true_hav_alt)
        return result

    def _freeze_attitude_after(self, attitude: np.ndarray, time: np.ndarray, freeze_time: float) -> np.ndarray:
        idx = np.searchsorted(time, freeze_time)
        attitude[idx:] = attitude[idx - 1]
        return attitude

    def _normalize_quaternion_sign(self, attitude) -> np.ndarray:
        """Force qw >= 0 by negating quaternions with a negative scalar part."""
        result = attitude.copy()
        mask = result[:, 0] < 0
        result[mask] = -result[mask]
        return result
