import numpy as np

from pipeline.dropout.core.dropout_strategy import WindowDropout, RandomDropout, ExternalSignalThresholdDropout

RNG = np.random.default_rng(0)


class TestWindowDropout:
    def test_samples_inside_window_are_dropped(self):
        time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        strategy = WindowDropout(windows=[(1.5, 2.5)])
        mask = strategy.generate_mask(time, RNG)
        assert mask.tolist() == [False, False, True, False, False]

    def test_endpoints_are_inclusive(self):
        time = np.array([0.5, 1.0, 2.0, 3.0, 3.5])
        strategy = WindowDropout(windows=[(1.0, 3.0)])
        mask = strategy.generate_mask(time, RNG)
        assert mask.tolist() == [False, True, True, True, False]

    def test_multiple_windows_combined_with_or(self):
        time = np.arange(10, dtype=float)
        strategy = WindowDropout(windows=[(1.0, 2.0), (7.0, 8.0)])
        mask = strategy.generate_mask(time, RNG)
        assert mask[1] and mask[2]
        assert mask[7] and mask[8]
        assert not mask[0] and not mask[5] and not mask[9]

    def test_no_samples_in_window_drops_nothing(self):
        time = np.array([0.0, 0.5, 1.0])
        strategy = WindowDropout(windows=[(5.0, 10.0)])
        mask = strategy.generate_mask(time, RNG)
        assert not mask.any()


class TestRandomDropout:
    def test_probability_zero_drops_nothing(self):
        time = np.linspace(0, 1, 1000)
        strategy = RandomDropout(probability=0.0)
        mask = strategy.generate_mask(time, np.random.default_rng(0))
        assert not mask.any()

    def test_probability_one_drops_everything(self):
        time = np.linspace(0, 1, 100)
        strategy = RandomDropout(probability=1.0)
        mask = strategy.generate_mask(time, np.random.default_rng(0))
        assert mask.all()

    def test_drop_rate_close_to_probability(self):
        time = np.linspace(0, 10, 10_000)
        strategy = RandomDropout(probability=0.3)
        mask = strategy.generate_mask(time, np.random.default_rng(42))
        assert abs(mask.mean() - 0.3) < 0.02

    def test_same_seed_produces_same_mask(self):
        time = np.linspace(0, 1, 200)
        strategy = RandomDropout(probability=0.4)
        mask_a = strategy.generate_mask(time, np.random.default_rng(7))
        mask_b = strategy.generate_mask(time, np.random.default_rng(7))
        np.testing.assert_array_equal(mask_a, mask_b)


class TestExternalSignalThresholdDropout:
    def _make_strategy(self, ref_values, threshold):
        ref_time = np.arange(len(ref_values), dtype=float)
        return ExternalSignalThresholdDropout(ref_time, np.array(ref_values, dtype=float), threshold)

    def test_samples_above_threshold_are_dropped(self):
        strategy = self._make_strategy([1.0, 5.0, 2.0, 6.0, 1.0], threshold=3.0)
        time = np.arange(5, dtype=float)
        mask = strategy.generate_mask(time, RNG)
        assert mask.tolist() == [False, True, False, True, False]

    def test_samples_at_threshold_are_not_dropped(self):
        strategy = self._make_strategy([3.0, 3.0, 3.0], threshold=3.0)
        time = np.arange(3, dtype=float)
        mask = strategy.generate_mask(time, RNG)
        assert not mask.any()

    def test_nothing_dropped_when_all_below_threshold(self):
        strategy = self._make_strategy([1.0, 2.0, 0.5], threshold=10.0)
        time = np.arange(3, dtype=float)
        mask = strategy.generate_mask(time, RNG)
        assert not mask.any()

    def test_everything_dropped_when_all_above_threshold(self):
        strategy = self._make_strategy([5.0, 6.0, 7.0], threshold=4.0)
        time = np.arange(3, dtype=float)
        mask = strategy.generate_mask(time, RNG)
        assert mask.all()

    def test_interpolates_onto_sensor_time_grid(self):
        # reference at t=0→0, t=2→10: midpoint t=1 should interpolate to 5, above threshold=3
        ref_time = np.array([0.0, 2.0])
        ref_values = np.array([0.0, 10.0])
        strategy = ExternalSignalThresholdDropout(ref_time, ref_values, threshold=3.0)
        sensor_time = np.array([0.0, 1.0, 2.0])
        mask = strategy.generate_mask(sensor_time, RNG)
        assert mask.tolist() == [False, True, True]
