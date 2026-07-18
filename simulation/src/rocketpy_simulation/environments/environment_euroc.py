from dataclasses import dataclass
from pathlib import Path

from rocketpy import Environment

from ..base import EnvironmentBase

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class EuroCEnvironment(EnvironmentBase):
    nc_file_path: str = str(_DATA_DIR / "EuroC_pressure_levels_reanalysis_2021-2024.nc")
    launch_date: tuple = (2023, 10, 14, 12)  # (year, month, day, hour UTC)
    latitude: float = 39.389700
    longitude: float = -8.288964
    elevation: float = 113  # meters above sea level
    max_expected_height: float = 10_000  # meters

    def build(self) -> Environment:
        env = Environment(
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            timezone="Europe/Portugal",
            datum="WGS84",
        )
        env.set_date(self.launch_date)
        env.set_atmospheric_model(
            type="Reanalysis",
            file=self.nc_file_path,
            dictionary="ECMWF",
        )
        env.max_expected_height = self.max_expected_height
        return env
