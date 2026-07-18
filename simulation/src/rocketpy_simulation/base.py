from abc import ABC, abstractmethod

from rocketpy import Environment, Motor, Rocket


class MotorBase(ABC):
    @abstractmethod
    def build(self) -> Motor: ...


class RocketBase(ABC):
    @abstractmethod
    def build(self) -> Rocket: ...


class EnvironmentBase(ABC):
    latitude: float
    longitude: float
    elevation: float

    @abstractmethod
    def build(self) -> Environment: ...
