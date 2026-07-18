from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd


class SensorOrigin(Enum):
    """How a sensor stream was produced."""

    SIMULATED_CLEAN = auto()
    """Clean output straight from the physics engine (no noise)."""

    SIMULATED_NOISY = auto()
    """Physics engine output with simulated noise by the physics engine"""

    SYNTHETIC = auto()
    """Derived from ground truth (data that can't be produced directly by physics engine)"""

    PIPELINE_PROCESSED = auto()
    """Pipeline altered noise data (noise injection, dropout, ect.)"""


@dataclass
class SensorStream:
    """A single sensor's time-series output.

    Stores timestamps and (multi-channel) data of simulated sensors.
    This component acts as the adapter between the physics engine and the rest of the pipeline.
    """

    name: str
    time: np.ndarray  # shape: (samples,)
    data: np.ndarray  # shape: (samples, channels) -> always stores 2-D data, even if values are 1-D
    channels: list[str]
    unit: str
    sampling_rate: float
    origin: SensorOrigin = SensorOrigin.SIMULATED_CLEAN
    processing_history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.data.ndim == 1:
            # reshape 1-D array to 2-D (like [[1], [2], [3]] -> for ease of use further down the line
            self.data = self.data.reshape(-1, 1)
        self.validate()

    @property
    def dt(self) -> float:
        """Sampling interval in seconds."""
        return 1.0 / self.sampling_rate

    @property
    def n_samples(self) -> int:
        return len(self.time)

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def sensor_data_shape(self) -> tuple[int, int]:
        return self.data.shape

    def copy(self) -> SensorStream:
        """Performs a deep copy (returns a new object, original data is not affected) if validation passes."""
        return SensorStream(
            name=self.name,
            time=self.time.copy(),
            data=self.data.copy(),
            channels=list(self.channels),
            unit=self.unit,
            sampling_rate=self.sampling_rate,
            origin=self.origin,
            processing_history=list(self.processing_history),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export as a DataFrame with a `time` column plus one column per channel."""
        df = pd.DataFrame(self.data, columns=self.channels)
        df.insert(0, "time", self.time)
        return df

    def add_processing_history_entry(self, entry: str) -> None:
        self.processing_history.append(entry)

    def validate(self) -> None:
        """Validate that all fields are consistent"""
        # Type checks
        if not isinstance(self.time, np.ndarray):
            raise TypeError(f"time must be np.ndarray, got {type(self.time).__name__}")
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data must be np.ndarray, got {type(self.data).__name__}")

        # Dimensionality
        if self.time.ndim != 1:
            raise ValueError(f"time must be 1-D, got {self.time.ndim}-D")
        if self.data.ndim != 2:
            raise ValueError(
                f"data must be 2-D, got {self.data.ndim}-D")  # also true for 1-D data, this is stored as [[1], [2]]

        # Shape
        if self.data.shape[0] != self.time.shape[0]:
            raise ValueError(
                f"data has {self.data.shape[0]} rows but time has {self.time.shape[0]} entries"
            )
        if self.data.shape[1] != len(self.channels):
            raise ValueError(
                f"data has {self.data.shape[1]} columns but "
                f"{len(self.channels)} channel names were given"
            )

        # Channel names
        if len(self.channels) == 0:
            raise ValueError("channels must not be empty")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError(f"channel names must be unique, got {self.channels}")

        # Sampling rate
        if self.sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {self.sampling_rate}")

        # Time
        if self.n_samples > 1 and not np.all(np.diff(self.time) > 0):
            raise ValueError("time must be strictly monotonically increasing")

        # Finite values
        if not np.all(np.isfinite(self.time)):
            raise ValueError("time contains NaN or Inf values")
        if self.origin is not SensorOrigin.PIPELINE_PROCESSED:
            if not np.all(np.isfinite(self.data)):
                raise ValueError(
                    f" data contains NaN or Inf values but origin is {self.origin.name}"
                )

        print(f"SensorStream: validation for sensor {self.name} passed!")
