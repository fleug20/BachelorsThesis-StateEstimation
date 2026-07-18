import numpy as np

from core.sensor_stream import SensorOrigin, SensorStream
from pipeline.noise.core.noise_profile import NoiseProfile
from pipeline.noise.noise_injector import NoiseInjector


def _noisy_profile(name="noisy") -> NoiseProfile:
    return NoiseProfile(
        name=name,
        unit="m/s²",
        noise_density=0.1,
        bias_fixed=0.0,
        random_walk=0.001,
        scale_factor_error=0.0,
        range_min=-1e10,
        range_max=1e10,
        resolution=None,
    )


def _zero_profile(name="zero") -> NoiseProfile:
    return NoiseProfile(
        name=name,
        unit="m/s²",
        noise_density=0.0,
        bias_fixed=0.0,
        random_walk=0.0,
        scale_factor_error=0.0,
        range_min=-1e10,
        range_max=1e10,
        resolution=None,
    )


class TestNoiseInjector:
    def test_noise_changes_signal(self, make_stream):
        stream = make_stream(np.zeros(500))
        injector = NoiseInjector(_noisy_profile(), seed=0)
        out = injector.apply(stream)
        assert not np.allclose(out.data, stream.data)

    def test_zero_noise_leaves_signal_unchanged(self, make_stream):
        data = np.linspace(0, 10, 200)
        stream = make_stream(data)
        injector = NoiseInjector(_zero_profile(), seed=0)
        out = injector.apply(stream)
        np.testing.assert_allclose(out.data[:, 0], data)

    def test_output_origin_is_pipeline_processed(self, make_stream):
        stream = make_stream(np.ones(100))
        out = NoiseInjector(_noisy_profile(), seed=0).apply(stream)
        assert out.origin == SensorOrigin.PIPELINE_PROCESSED

    def test_processing_history_updated(self, make_stream):
        stream = make_stream(np.ones(100))
        n_before = len(stream.processing_history)
        out = NoiseInjector(_noisy_profile(), seed=0).apply(stream)
        assert len(out.processing_history) == n_before + 1

    def test_does_not_mutate_input(self, make_stream):
        data = np.ones(100)
        stream = make_stream(data.copy())
        original = stream.data.copy()
        NoiseInjector(_noisy_profile(), seed=0).apply(stream)
        np.testing.assert_array_equal(stream.data, original)

    def test_channel_selection_only_affects_target(self, make_multi_stream):
        data = np.ones((300, 3))
        stream = make_multi_stream(data)
        injector = NoiseInjector(_noisy_profile(), seed=0, channels=[1])
        out = injector.apply(stream)
        assert not np.allclose(out.data[:, 1], 1.0)
        np.testing.assert_allclose(out.data[:, 0], 1.0)
        np.testing.assert_allclose(out.data[:, 2], 1.0)

    def test_same_seed_gives_identical_output(self, make_stream):
        stream = make_stream(np.zeros(300))
        out_a = NoiseInjector(_noisy_profile(), seed=7).apply(stream)
        out_b = NoiseInjector(_noisy_profile(), seed=7).apply(stream)
        np.testing.assert_array_equal(out_a.data, out_b.data)

    def test_different_seeds_give_different_output(self, make_stream):
        stream = make_stream(np.zeros(300))
        out_a = NoiseInjector(_noisy_profile(), seed=1).apply(stream)
        out_b = NoiseInjector(_noisy_profile(), seed=2).apply(stream)
        assert not np.allclose(out_a.data, out_b.data)


def _gnss_stream(n: int = 200, sampling_rate: float = 20.0) -> SensorStream:
    time = np.arange(n, dtype=float) / sampling_rate
    data = np.column_stack([
        np.full(n, 39.39),  # latitude
        np.full(n, -8.29),  # longitude
        np.linspace(113, 500, n),  # altitude
    ])
    return SensorStream(
        name="gnss",
        time=time,
        data=data,
        channels=["latitude", "longitude", "altitude"],
        unit="deg/deg/m",
        sampling_rate=sampling_rate,
        origin=SensorOrigin.SIMULATED_CLEAN,
    )
