from dataclasses import dataclass
from pathlib import Path

from rocketpy import Rocket, Motor, Accelerometer, Gyroscope, Barometer, GnssReceiver

from ..base import RocketBase

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class AsteriaRocket(RocketBase):
    motor: Motor
    drag_curve_path: str = str(_DATA_DIR / "DragCurve.csv")
    include_parachutes: bool = True

    # Rocket
    # NOTE: using `tail_to_nose` coordinate system due to incorrect sensor data when using `nose_to_tail`
    radius: float = 0.0895  # m
    mass: float = 46.01  # kg (dry mass without motor) -> total dry mass of 49 kg
    inertia: tuple = (94.51, 94.52, 0.2107)
    center_of_mass_without_motor: float = -2.90  # m
    motor_position: float = -5.32  # m
    nose_length: float = 0.81  # m
    sensor_position: float = -2.978  # m -> actual CoM (dry mass including motor) to minimize lever arm effect

    # Rail buttons
    upper_button_position: float = -1.83116903  # m
    lower_button_position: float = -5.0  # m

    # Fins
    fin_n: int = 4
    fin_root_chord: float = 0.20272964694030443  # m
    fin_tip_chord: float = 0.11889239635534514  # m
    fin_span: float = 0.20848681179344536 - 0.0005500858676339606  # m
    fin_position: float = -4.708601669707605  # m
    fin_cant_angle: float = 0.5  # degrees

    # Tail
    tail_top_radius: float = 0.0895  # m
    tail_bottom_radius: float = 0.08295  # m
    tail_length: float = 0.38  # m
    tail_position: float = -5.0  # m

    # Drogue parachute
    drogue_cd_s: float = 0.7 * 1.0  # Cd × area (m^2)
    drogue_lag: float = 1.5  # s
    drogue_noise: tuple = (0, 8.3, 0.5)

    # Main parachute
    main_cd_s: float = 2.2 * 11.0  # Cd × area (m^2)
    main_trigger_altitude: float = 2000.0  # m AGL
    main_lag: float = 1.5  # s
    main_noise: tuple = (0, 8.3, 0.5)

    # Accelerometer
    accelerometer_position = sensor_position
    accelerometer_orientation = (0, 0, 0)
    accelerometer_sampling_rate: int = 800  # Hz

    # Gyroscope
    gyroscope_position = sensor_position
    gyroscope_orientation = (0, 0, 0)
    gyroscope_sampling_rate: int = 800  # Hz

    # Barometer
    barometer_position = sensor_position
    barometer_sampling_rate: int = 100  # Hz

    # GNSS Receiver
    gnss_position = sensor_position
    gnss_sampling_rate: int = 25  # Hz
    horizontal_position_accuracy: float = 1.5  # m
    vertical_position_accuracy: float = 2.25  # m

    def __post_init__(self):
        self.gnss_clean = GnssReceiver(
            sampling_rate=self.gnss_sampling_rate,
            position_accuracy=0.0,
            altitude_accuracy=0.0,
            name="gnss_clean"
        )
        self.gnss = GnssReceiver(
            sampling_rate=self.gnss_sampling_rate,
            position_accuracy=self.horizontal_position_accuracy,
            altitude_accuracy=self.vertical_position_accuracy,
            name="gnss"
        )
        self.gyroscope_clean = Gyroscope(
            orientation=self.gyroscope_orientation,
            sampling_rate=self.gyroscope_sampling_rate,
            name="gyroscope_clean",
            noise_density=0,
        )
        self.barometer_clean = Barometer(
            sampling_rate=self.barometer_sampling_rate,
            name="barometer_clean",
            noise_density=0,
        )
        self.accelerometer_clean = Accelerometer(
            sampling_rate=self.accelerometer_sampling_rate,
            orientation=self.accelerometer_orientation,
            noise_density=0,
            random_walk_density=0,
            consider_gravity=False,
            name="accelerometer_clean",
        )

    def build(self) -> Rocket:
        rocket = Rocket(
            radius=self.radius,
            mass=self.mass,
            inertia=self.inertia,
            power_off_drag=self.drag_curve_path,
            power_on_drag=self.drag_curve_path,
            center_of_mass_without_motor=self.center_of_mass_without_motor,
            coordinate_system_orientation="tail_to_nose",
        )

        rocket.add_motor(self.motor, position=self.motor_position)

        rocket.set_rail_buttons(
            upper_button_position=self.upper_button_position,
            lower_button_position=self.lower_button_position,
            angular_position=0.0,
        )

        rocket.add_nose(
            length=self.nose_length,
            kind="Von Karman",
            position=0.0,
        )

        rocket.add_trapezoidal_fins(
            n=self.fin_n,
            root_chord=self.fin_root_chord,
            tip_chord=self.fin_tip_chord,
            span=self.fin_span,
            position=self.fin_position,
            cant_angle=self.fin_cant_angle,
        )

        rocket.add_tail(
            top_radius=self.tail_top_radius,
            bottom_radius=self.tail_bottom_radius,
            length=self.tail_length,
            position=self.tail_position,
        )

        if self.include_parachutes:
            rocket.add_parachute(
                name="Drogue",
                cd_s=self.drogue_cd_s,
                trigger="apogee",
                sampling_rate=100,
                lag=self.drogue_lag,
                noise=self.drogue_noise,
            )

            rocket.add_parachute(
                name="Main",
                cd_s=self.main_cd_s,
                trigger=self.main_trigger_altitude,
                sampling_rate=100,
                lag=self.main_lag,
                noise=self.main_noise,
            )

        rocket.add_sensor(self.accelerometer_clean, position=self.accelerometer_position)
        rocket.add_sensor(self.gyroscope_clean, position=self.gyroscope_position)
        rocket.add_sensor(self.barometer_clean, position=self.barometer_position)
        rocket.add_sensor(self.gnss_clean, position=self.gnss_position)
        rocket.add_sensor(self.gnss, position=self.gnss_position)

        return rocket
