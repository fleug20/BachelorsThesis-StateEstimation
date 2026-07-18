import numpy as np

from pipeline.noise.core.noise_profile import NoiseProfile
from pipeline.noise.core.sensor_model import SensorModel

DT = 0.01  # 100 Hz


def _profile(**overrides) -> NoiseProfile:
    defaults = dict(
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
    defaults.update(overrides)
    return NoiseProfile(**defaults)


class TestSensorModelConstantBias:
    def test_bias_fixed_shifts_signal(self):
        signal = np.ones(100) * 5.0
        model = SensorModel(_profile(bias_fixed=2.0), seed=0)
        out = model.apply(signal.copy(), DT)
        np.testing.assert_allclose(out, 5.0 + model.bias_fixed)

    def test_bias_fixed_is_within_3sigma_bound(self):
        bound = 2.0
        n_samples = 1000
        within_bound = 0
        for seed in range(n_samples):
            model = SensorModel(_profile(bias_fixed=bound), seed=seed)
            if abs(model.bias_fixed) <= bound:
                within_bound += 1
        assert within_bound / n_samples >= 0.99


class TestSensorModelScaleFactor:
    def test_scale_factor_stretches_signal(self):
        signal = np.ones(100) * 10.0
        model = SensorModel(_profile(scale_factor_error=0.1), seed=0)
        out = model.apply(signal.copy(), DT)
        np.testing.assert_allclose(out, 11.0)

    def test_negative_scale_factor_compresses_signal(self):
        signal = np.ones(100) * 10.0
        model = SensorModel(_profile(scale_factor_error=-0.1), seed=0)
        out = model.apply(signal.copy(), DT)
        np.testing.assert_allclose(out, 9.0)


class TestSensorModelClipping:
    def test_values_clipped_to_range(self):
        signal = np.array([0.0, 5.0, -5.0, 10.0, -10.0])
        model = SensorModel(_profile(range_min=-3.0, range_max=3.0), seed=0)
        out = model.apply(signal.copy(), DT)
        assert out.max() <= 3.0
        assert out.min() >= -3.0

    def test_in_range_values_unchanged(self):
        signal = np.array([1.0, -1.0, 0.0])
        model = SensorModel(_profile(range_min=-2.0, range_max=2.0), seed=0)
        out = model.apply(signal.copy(), DT)
        np.testing.assert_allclose(out, signal)


class TestSensorModelResolution:
    def test_output_is_multiple_of_resolution(self):
        signal = np.array([1.1, 2.4, 3.7, -0.6])
        resolution = 0.5
        model = SensorModel(_profile(resolution=resolution), seed=0)
        out = model.apply(signal.copy(), DT)
        remainders = out % resolution
        # remainders should be 0 (or very close due to floating point)
        np.testing.assert_allclose(remainders, 0.0, atol=1e-10)

    def test_none_resolution_skips_quantization(self):
        signal = np.array([1.111, 2.222])
        model = SensorModel(_profile(resolution=None), seed=0)
        out = model.apply(signal.copy(), DT)
        # without quantization, non-round values survive
        assert out[0] != round(out[0])


class TestSensorModelWhiteNoise:
    def test_white_noise_adds_variation(self):
        signal = np.ones(1000)
        model = SensorModel(_profile(noise_density=0.1), seed=0)
        out = model.apply(signal.copy(), DT)
        assert out.std() > 0.0

    def test_zero_noise_density_no_variation(self):
        signal = np.ones(100)
        model = SensorModel(_profile(noise_density=0.0), seed=0)
        out = model.apply(signal.copy(), DT)
        np.testing.assert_allclose(out, 1.0)


class TestSensorModelReproducibility:
    def test_same_seed_produces_identical_output(self):
        signal = np.random.default_rng(1).normal(size=200)
        model_a = SensorModel(_profile(noise_density=0.01, random_walk=0.001), seed=42)
        model_b = SensorModel(_profile(noise_density=0.01, random_walk=0.001), seed=42)
        np.testing.assert_array_equal(
            model_a.apply(signal.copy(), DT),
            model_b.apply(signal.copy(), DT),
        )

    def test_different_seeds_produce_different_output(self):
        signal = np.ones(200)
        out_a = SensorModel(_profile(noise_density=0.1), seed=1).apply(signal.copy(), DT)
        out_b = SensorModel(_profile(noise_density=0.1), seed=2).apply(signal.copy(), DT)
        assert not np.allclose(out_a, out_b)
