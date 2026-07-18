from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class NoiseProfile:
    name: str
    unit: str
    noise_density: float  # white noise density. Higher is worse
    bias_fixed: float  # fixed offset
    random_walk: float  # slow drift of the bias over time
    scale_factor_error: float  # if sensors output is slightly stretched or compressed
    range_min: float  # minimum value. Data is capped to this value
    range_max: float  # max value. Data is capped to this value
    resolution: float | None = None  # smallest possible change (determined by the bit-size of the value)

    @classmethod
    def from_yaml(cls, path: str | Path) -> NoiseProfile:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        for field in ("noise_density", "bias_fixed", "random_walk",
                      "scale_factor_error", "range_min", "range_max", "resolution"):
            if field in data and data[field] is not None:
                data[field] = float(data[field])
        return cls(**data)
