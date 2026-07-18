import numpy as np
import pytest

from core.sensor_stream import SensorStream, SensorOrigin
from pipeline.noise.core.noise_profile import NoiseProfile


@pytest.fixture
def make_stream():
    """Factory for a simple single-channel SensorStream."""
    def _make(data: np.ndarray, sampling_rate: float = 100.0, name: str = "test") -> SensorStream:
        n = len(data)
        time = np.arange(n, dtype=float) / sampling_rate
        data_2d = data.reshape(-1, 1).astype(float)
        return SensorStream(
            name=name,
            time=time,
            data=data_2d,
            channels=["x"],
            unit="m",
            sampling_rate=sampling_rate,
            origin=SensorOrigin.SIMULATED_CLEAN,
        )
    return _make


@pytest.fixture
def make_multi_stream():
    """Factory for a 3-channel SensorStream."""
    def _make(data: np.ndarray, sampling_rate: float = 100.0, name: str = "test") -> SensorStream:
        assert data.ndim == 2 and data.shape[1] == 3
        n = data.shape[0]
        time = np.arange(n, dtype=float) / sampling_rate
        return SensorStream(
            name=name,
            time=time,
            data=data.astype(float),
            channels=["x", "y", "z"],
            unit="m",
            sampling_rate=sampling_rate,
            origin=SensorOrigin.SIMULATED_CLEAN,
        )
    return _make


@pytest.fixture
def zero_profile():
    return NoiseProfile(
        name="zero",
        unit="m/s²",
        noise_density=0.0,
        bias_fixed=0.0,
        random_walk=0.0,
        scale_factor_error=0.0,
        range_min=-1e10,
        range_max=1e10,
        resolution=None,
    )
