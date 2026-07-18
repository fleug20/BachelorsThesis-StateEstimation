from dataclasses import dataclass
from pathlib import Path

from rocketpy import GenericMotor

from ..base import MotorBase

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class AsteriaMotor(MotorBase):
    thrust_curve_path: str = str(
        _DATA_DIR / "ThrustCurve_optimized.csv")  # optimized thrust curve (scaled by factor 1.15)
    coordinate_system_orientation: str = "nozzle_to_combustion_chamber"

    # Dry structure
    dry_mass: float = 2.99  # kg
    dry_inertia: tuple = (65.035, 64.931, 0.7440)

    # Nozzle geometry
    nozzle_radius: float = 0.051115125  # m
    throat_radius: float = 0.034076749999999996  # m
    nozzle_position: float = 0.0

    # Chamber geometry
    chamber_radius: float = 0.0845  # m (tank radius)
    chamber_height: float = 0.520 + 0.620 + 0.50051981  # m (sum of tank heights)
    chamber_position: float = 1.07  # m

    # Propellant masses
    lox_density: float = 1141.0  # kg/m^3
    lox_volume: float = 0.0131  # m^3
    eth_density: float = 789.0  # kg/m^3
    eth_volume: float = 0.0131  # m^3
    pressurant_density: float = 303.0  # kg/m^3 (at ~300 bar)
    pressurant_volume: float = 0.0068  # m^3

    def _propellant_mass(self) -> float:
        lox = self.lox_density * self.lox_volume
        eth = self.eth_density * self.eth_volume
        pressurant = self.pressurant_density * self.pressurant_volume
        return lox + eth + pressurant

    def build(self) -> GenericMotor:
        return GenericMotor(
            thrust_source=self.thrust_curve_path,
            dry_mass=self.dry_mass,
            dry_inertia=self.dry_inertia,
            nozzle_radius=self.nozzle_radius,
            center_of_dry_mass_position=0,
            nozzle_position=self.nozzle_position,
            burn_time=None,  # inferred from thrust curve
            chamber_radius=self.chamber_radius,
            chamber_height=self.chamber_height,
            chamber_position=self.chamber_position,
            propellant_initial_mass=self._propellant_mass(),
            coordinate_system_orientation=self.coordinate_system_orientation,
            reshape_thrust_curve=False,
            interpolation_method="linear"
        )
