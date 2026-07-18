from core import SimulationResult
from pipeline.dropout import DropoutInjector
from pipeline.noise import NoiseInjector
from pipeline.synthetic import SyntheticSensor


class SyntheticSensorGenerator:
    """Creates synthetic sensor streams from ground truth.

    Each entry is a (sensor, noise_injector, dropout_injector) tuple.
    A clean stream is always stored. If a noise injector is provided,
    a noisy copy is generated and stored as well.
    If a dropout injector is provided, the noisy stream is passed through it.
    """

    def __init__(self, sensors: list[tuple[SyntheticSensor, NoiseInjector | None, DropoutInjector | None]]):
        self._sensors = sensors

    def apply(self, result: SimulationResult) -> SimulationResult:
        for sensor, injector, dropout in self._sensors:
            clean_stream = sensor.generate(result.ground_truth)

            clean_name = f"{clean_stream.name}_clean"
            clean_stream.name = clean_name
            result.add_sensor_clean(clean_name, clean_stream)

            if injector is None and dropout is None:
                continue

            noisy_stream = clean_stream  # no copy -> but this will happen in the apply methods!
            if injector is not None:
                noisy_stream = injector.apply(noisy_stream)
            if dropout is not None:
                noisy_stream = dropout.apply(noisy_stream)
            noisy_stream.name = sensor.name

            result.add_sensor_noisy(sensor.name, noisy_stream)

        return result
