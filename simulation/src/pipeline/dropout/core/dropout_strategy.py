from abc import ABC, abstractmethod

import numpy as np


class DropoutStrategy(ABC):
    """Strategy object that generates a boolean dropout mask for a time series.

    A mask entry of `True` means: drop this sample. Multiple events are
    combined with logical OR by the DropoutInjector, so any event can trigger
    a drop independently of the others.
    """

    @abstractmethod
    def generate_mask(self, time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        pass

    def describe(self) -> str:
        return type(self).__name__


class WindowDropout(DropoutStrategy):
    """Drops every sample whose timestamp falls inside any of the given
    (start, end) windows. Timestamps are in seconds (matching time channel).
    Both endpoints are inclusive."""

    def __init__(self, windows: list[tuple[float, float]]):
        if not windows:
            raise ValueError("WindowDropout: windows must not be empty")
        for i, window in enumerate(windows):
            start, end = window
            if end < start:
                raise ValueError(
                    f"WindowDropout: window {i} end ({end}) must be >= start ({start})"
                )
        self.windows = windows

    def generate_mask(self, time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mask = np.zeros(len(time), dtype=bool)
        for start, end in self.windows:
            mask |= (time >= start) & (time <= end)
        return mask

    def describe(self) -> str:
        windows_str = ", ".join(f"({s}s, {e}s)" for s, e in self.windows)
        return f"WindowDropout(windows=[{windows_str}])"


class RandomDropout(DropoutStrategy):
    """Independent drop chance on each sample."""

    def __init__(self, probability: float):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"RandomDropout: probability must be in [0, 1], got {probability}")
        self.probability = float(probability)

    def generate_mask(self, time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.random(len(time)) < self.probability

    def describe(self) -> str:
        return f"RandomDropout(p={self.probability})"


class ExternalSignalThresholdDropout(DropoutStrategy):
    """Drops samples where a reference signal exceeds a threshold.

    The reference signal is interpolated onto the sensor's time grid at
    mask-generation time. Useful for e.g. dropping GNSS above a g-load limit.

    Args:
        reference_time:   Time axis of the reference signal (seconds).
        reference_values: Scalar values aligned with reference_time.
        threshold:        Samples where interpolated value > threshold are dropped.
    """

    def __init__(self, reference_time: np.ndarray, reference_values: np.ndarray, threshold: float):
        self.reference_time = reference_time
        self.reference_values = reference_values
        self.threshold = float(threshold)

    def generate_mask(self, time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        interpolated = np.interp(time, self.reference_time, self.reference_values)
        return interpolated > self.threshold

    def describe(self) -> str:
        return f"ExternalSignalThresholdDropout(threshold={self.threshold})"
