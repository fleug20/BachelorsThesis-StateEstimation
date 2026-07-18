import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .ground_truth import GroundTruth
from .sensor_stream import SensorStream


@dataclass
class SimulationResult:
    """output of a single simulation run (independent of simulator).

    This is the data contract between simulators (RocketPy, RotorPy, …)
    and the downstream pipeline (noise injection, dropout, evaluation, ...).
    """

    ground_truth: GroundTruth
    sensors_clean: dict[str, SensorStream] = field(default_factory=dict)
    sensors_noisy: dict[str, SensorStream] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate_all()

    def sensor_names(self) -> list[str]:
        return list({**self.sensors_clean, **self.sensors_noisy}.keys())

    def add_sensor_clean(self, name: str, stream: SensorStream) -> None:
        self.sensors_clean[name] = stream

    def add_sensor_noisy(self, name: str, stream: SensorStream) -> None:
        self.sensors_noisy[name] = stream

    def export_csv_data(self, output_dir: str) -> list[str]:
        """Write each sensor stream to `<output_dir>/<name>.csv` as well as a yaml file containing the metadata."""
        self.validate_all()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # export sensors
        paths: list[str] = []
        for name, stream in {**self.sensors_clean, **self.sensors_noisy}.items():
            p = out / f"{name}.csv"
            stream.to_dataframe().to_csv(p, index=False)
            paths.append(str(p))

        # update metadata
        self.metadata["sensors_clean"] = list(self.sensors_clean.keys())
        self.metadata["sensors_noisy"] = list(self.sensors_noisy.keys())

        sensor_meta: dict[str, dict] = {}
        for name, stream in {**self.sensors_clean, **self.sensors_noisy}.items():
            sensor_meta[name] = {
                "name": stream.name,
                "unit": stream.unit,
                "sampling_rate": stream.sampling_rate,
                "origin": stream.origin.name,
                "processing_history": stream.processing_history,
            }
        self.metadata["sensors"] = sensor_meta

        # export metadata
        path = out / "metadata.yaml"
        with open(path, "w") as f:
            yaml.dump(self.metadata, f, default_flow_style=False)
            paths.append(str(path))

        gt_path = out / "ground_truth.csv"
        self.ground_truth.to_dataframe().to_csv(gt_path, index=False)
        paths.append(str(gt_path))
        return paths

    def validate_all(self):
        self.ground_truth.validate()
        for stream in self.sensors_clean.values():
            stream.validate()
        for stream in self.sensors_noisy.values():
            stream.validate()

    def save(self, filepath: str) -> None:
        self.validate_all()
        path = Path(filepath)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> SimulationResult:
        path = Path(filepath)
        with open(path, "rb") as f:
            return pickle.load(f)
