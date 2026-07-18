import datetime
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core import GroundTruth
from evaluation.core.kalman_estimate import KalmanEstimate
from evaluation.core.metric import MetricResult


@dataclass
class EvaluationResult:
    """Output of one evaluation run."""

    ground_truth: GroundTruth
    estimate: KalmanEstimate
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate_all()

    def summary(self) -> None:
        print("=" * 60)
        name = self.metadata['simulation_metadata']['name']
        print(f"Evaluation Result, {name if name else datetime.datetime.now()}")
        print("=" * 60)

        print(f"number of samples: {self.estimate.time.shape[0]}")
        print("\nMetrics:")
        for metric_name, result in self.metrics.items():
            print(f"  {metric_name:<30} {result.value:>12.6f} {result.unit}")
            if result.per_axis:
                for axis, v in result.per_axis.items():
                    print(f"    .{axis:<26} {v:>12.6f} {result.unit}")

    def to_yaml(self, filepath: str | Path) -> None:
        """only includes metadata and metrics, no gt or estimate"""
        payload: dict[str, Any] = {
            "metadata": self.metadata,
            "metrics": {n: self._result_to_dict(r) for n, r in self.metrics.items()},
        }
        with open(filepath, "w") as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=False)

    def save(self, filepath: str | Path) -> None:
        self.validate_all()
        path = Path(filepath)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def validate_all(self):
        self.ground_truth.validate()
        self.estimate.validate()

    @classmethod
    def load(cls, filepath: str | Path) -> EvaluationResult:
        path = Path(filepath)
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _result_to_dict(result: MetricResult) -> dict[str, Any]:
        return {
            "value": result.value,
            "unit": result.unit,
            "per_axis": result.per_axis,
        }
