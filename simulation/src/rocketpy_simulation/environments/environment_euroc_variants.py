from dataclasses import dataclass, field

from rocketpy import Environment

from ..base import EnvironmentBase


@dataclass
class EuroCEnvironmentNoWind(EnvironmentBase):
    """ISA standard atmosphere with no wind"""

    launch_date: tuple = (2023, 10, 14, 12)
    latitude: float = 39.389700
    longitude: float = -8.288964
    elevation: float = 113
    max_expected_height: float = 10_000

    def build(self) -> Environment:
        env = Environment(
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            timezone="Europe/Portugal",
            datum="WGS84",
        )
        env.set_date(self.launch_date)
        env.set_atmospheric_model(type="standard_atmosphere")
        env.max_expected_height = self.max_expected_height
        return env


@dataclass
class EuroCEnvironmentSlightWind(EnvironmentBase):
    """Custom atmosphere with a light wind"""

    launch_date: tuple = (2023, 10, 14, 12)
    latitude: float = 39.389700
    longitude: float = -8.288964
    elevation: float = 113
    max_expected_height: float = 10_000
    wind_u: list = field(default_factory=lambda: [(0, 2.0), (1000, 3.0), (3000, 5.0), (10000, 6.0)])
    wind_v: list = field(default_factory=lambda: [(0, 0.5), (1000, 1.0), (3000, 1.5), (10000, 2.0)])

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
            type="custom_atmosphere",
            wind_u=self.wind_u,
            wind_v=self.wind_v,
        )
        env.max_expected_height = self.max_expected_height
        return env


@dataclass
class EuroCEnvironmentWindy(EnvironmentBase):
    """Custom atmosphere with a strong wind"""

    launch_date: tuple = (2023, 10, 14, 12)
    latitude: float = 39.389700
    longitude: float = -8.288964
    elevation: float = 113
    max_expected_height: float = 10_000
    wind_u: list = field(default_factory=lambda: [(0, 8.0), (1000, 12.0), (3000, 16.0), (10000, 20.0)])
    wind_v: list = field(default_factory=lambda: [(0, 3.0), (1000, 4.0), (3000, 6.0), (10000, 8.0)])

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
            type="custom_atmosphere",
            wind_u=self.wind_u,
            wind_v=self.wind_v,
        )
        env.max_expected_height = self.max_expected_height
        return env


@dataclass
class EuroCEnvironmentGust(EnvironmentBase):
    """Custom atmosphere with a sudden wind gust"""

    launch_date: tuple = (2023, 10, 14, 12)
    latitude: float = 39.389700
    longitude: float = -8.288964
    elevation: float = 113
    max_expected_height: float = 10_000

    wind_u: list = field(default_factory=lambda: [
        (0, 2.0), (1500, 3.0), (1600, 25.0), (1900, 25.0), (2000, 4.0), (10000, 6.0)
    ])
    wind_v: list = field(default_factory=lambda: [
        (0, 0.5), (1500, 1.0), (1600, 10.0), (1900, 10.0), (2000, 1.5), (10000, 2.0)
    ])

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
            type="custom_atmosphere",
            wind_u=self.wind_u,
            wind_v=self.wind_v,
        )
        env.max_expected_height = self.max_expected_height
        return env


@dataclass
class EuroCEnvironmentHotDay(EnvironmentBase):
    """Custom atmosphere: warm"""

    launch_date: tuple = (2023, 10, 14, 12)
    latitude: float = 39.389700
    longitude: float = -8.288964
    elevation: float = 113
    max_expected_height: float = 10_000
    temperature: float = 313.15  # 40 degrees Celsius
    pressure: float = 101325.0

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
            type="custom_atmosphere",
            wind_u=3.0,
            wind_v=0.0,
            temperature=self.temperature,
            pressure=self.pressure
        )
        env.max_expected_height = self.max_expected_height
        return env


@dataclass
class EuroCEnvironmentColdDay(EnvironmentBase):
    """Custom atmosphere: cold day"""

    launch_date: tuple = (2023, 1, 14, 12)  # Changed month to January for thematic accuracy
    latitude: float = 39.389700
    longitude: float = -8.288964
    elevation: float = 113
    max_expected_height: float = 10_000
    temperature: float = 273.15  # 0 degrees Celsius
    pressure: float = 101325.0

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
            type="custom_atmosphere",
            wind_u=0.0,
            wind_v=3.0,
            temperature=self.temperature,
            pressure=self.pressure
        )
        env.max_expected_height = self.max_expected_height
        return env
