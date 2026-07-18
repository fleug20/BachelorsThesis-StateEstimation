from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import scipy.constants
from scipy.spatial.transform import Rotation
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.environments import Environment
from rotorpy.sensors.imu import Imu
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.utils.postprocessing import unpack_sim_data
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.world import World

from core import GroundTruth, SensorStream, SimulationResult, SensorOrigin
from rotorpy_simulation import CrazyflieBase, IMUSensorBase

SENSOR_CHANNELS = {
    "accel": ["ax", "ay", "az"],
    "gyro": ["wx", "wy", "wz"],
}

SENSOR_UNITS = {
    "accel": "g",
    "gyro": "deg/s",
}


@dataclass
class CrazyflieSimulation:
    """Main class for running a crazyflie simulation with RotorPy.

    NOTE: Crazyflie sensors will be converted to units of "g" and "deg/s for accelerometer and gyroscope,
    but ground truth data will not be converted!
    """

    crazyflie_builder: CrazyflieBase = None
    imu_sensor_builder: IMUSensorBase = None

    # Flight parameters
    world: World = World.empty(extents=[-3, 3, -3, 3, 0, 3])
    trajectory: Any = ThreeDCircularTraj(center=np.array([0, 0, 2]),
                                         radius=np.array([2, 2, 0]),
                                         freq=np.array([0.2, 0.2, 0.2]))
    controller: Any = SE3Control(quad_params)

    # Simulation Parameters
    simulation_sampling_rate: float = 1000  # Hz
    simulation_duration: float = 15.0  # seconds
    simulation_output_name: str = field(default=None, init=False)

    # Simulation
    flight: Any = field(default=None, init=False)
    _crazyflie: Multirotor = field(default=None, init=False)
    _imu_sensor: Imu = field(default=None, init=False)

    def run(self, name: str) -> SimulationResult:
        print("CrazyflieSimulation: running simulation...")
        self._simulate()

        ground_truth: GroundTruth = self._get_ground_truth()
        clean, noisy = self._get_sensor_streams()
        metadata = {
            "simulator": "CrazyflieSimulation",
            "name": name,
            "simulation_datetime": datetime.now(),
            "simulation_sampling_rate": self.simulation_sampling_rate,
            "flight_time": self.simulation_duration,
        }

        return SimulationResult(ground_truth=ground_truth, sensors_clean=clean, sensors_noisy=noisy, metadata=metadata)

    def _simulate(self) -> None:
        self._crazyflie = self.crazyflie_builder.build()
        self._imu_sensor = self.imu_sensor_builder.build()
        sim_instance = Environment(vehicle=self._crazyflie,
                                   controller=self.controller,
                                   trajectory=self.trajectory,
                                   world=self.world,
                                   imu=self._imu_sensor,
                                   wind_profile=None,
                                   sim_rate=self.simulation_sampling_rate,
                                   safety_margin=0.25)
        self.flight = sim_instance.run(t_final=self.simulation_duration,
                                       use_mocap=False,
                                       terminate=False,
                                       plot=True,
                                       plot_mocap=False,
                                       plot_estimator=False,
                                       plot_imu=True,
                                       animate_bool=True,
                                       animate_wind=False,
                                       verbose=True,
                                       fname=self.simulation_output_name
                                       )

    def _get_ground_truth(self) -> GroundTruth:
        flight = self.flight
        df = unpack_sim_data(self.flight)

        time = flight["time"]  # (N,)
        position = flight["state"]["x"]
        velocity = flight["state"]["v"]  # (N, 3)
        acceleration = df[["ax_gt", "ay_gt", "az_gt"]].to_numpy()  # (N, 3)
        # WATCH OUT: rotorpy provides quaternions with scalar-last! Thus, manual reordering here
        attitude = df[["qw", "qx", "qy", "qz"]].to_numpy()  # (N, 4) [w, x, y, z]
        angular_velocity = flight["state"]["w"]  # (N, 3)

        nan1d = np.zeros_like(time)
        nan1d[nan1d == 0] = np.nan
        nan3d = np.zeros_like(position)
        nan3d[nan3d == 0] = np.nan
        pressure = nan1d  # (N, 3)

        return GroundTruth(
            time=time,
            position=position,
            geo_coordinates=nan3d,
            velocity=velocity,
            acceleration=acceleration,
            attitude=attitude,
            angular_velocity=angular_velocity,
            pressure=pressure,
            simulation_sampling_rate=self.simulation_sampling_rate
        )

    def _get_sensor_streams(self) -> tuple[dict[str, SensorStream], dict[str, SensorStream]]:
        clean: dict[str, SensorStream] = {}
        noisy: dict[str, SensorStream] = {}

        # imu_measurements (noisy values)
        for sensor in self.flight["imu_measurements"].items():
            sensor_type: str = sensor[0]
            name: str = sensor_type
            data: np.ndarray = sensor[1]

            # unit conversion (to mimic real units on the crazyflie)
            if sensor_type == "accel":
                data = data / scipy.constants.g  # m/s² -> G
            elif sensor_type == "gyro":
                data = np.rad2deg(data)  # rad/s -> °/s

            origin = SensorOrigin.SIMULATED_NOISY
            time = self.flight["time"]
            channels = SENSOR_CHANNELS[sensor_type]
            unit = SENSOR_UNITS[sensor_type]
            sampling_rate = self.imu_sensor_builder.sampling_rate
            processing_history = [
                f"Sensor data by RotorPy. Original name: {name}, Sensor origin: {origin}, Sampling rate: {str(sampling_rate)} Hz, Data units: {unit}"
            ]

            stream = SensorStream(name=name,
                                  time=time,
                                  data=data,
                                  channels=channels,
                                  unit=unit,
                                  sampling_rate=sampling_rate,
                                  origin=origin,
                                  processing_history=processing_history, )

            noisy[name] = stream

        time = self.flight["time"]
        sampling_rate = self.imu_sensor_builder.sampling_rate

        # Clean gyro:
        gyro_clean = np.rad2deg(self.flight["state"]["w"])

        # Clean accel:
        v_world = self.flight["state"]["v"]  # (N, 3) ENU
        q_WB = Rotation.from_quat(self.flight["state"]["q"])  # scalar-last [x,y,z,w]
        vdot_world = np.gradient(v_world, time, axis=0)  # (N, 3) central diff
        a_sf_world = vdot_world - np.array([0.0, 0.0, -9.81])  # specific force in ENU
        a_sf_body = q_WB.inv().apply(a_sf_world)  # rotate to body (FLU)
        accel_clean = a_sf_body / scipy.constants.g  # m/s² -> G

        for sensor_type, data in [("accel", accel_clean), ("gyro", gyro_clean)]:
            name = f"{sensor_type}_clean"
            origin = SensorOrigin.SIMULATED_CLEAN
            channels = SENSOR_CHANNELS[sensor_type]
            unit = SENSOR_UNITS[sensor_type]
            processing_history = [
                f"Clean IMU from ground truth state. Sensor origin: {origin}, "
                f"Sampling rate: {sampling_rate} Hz, Data units: {unit}"
            ]
            clean[name] = SensorStream(
                name=name,
                time=time,
                data=data,
                channels=channels,
                unit=unit,
                sampling_rate=sampling_rate,
                origin=origin,
                processing_history=processing_history,
            )

        return clean, noisy
