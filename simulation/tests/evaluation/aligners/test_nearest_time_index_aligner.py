import warnings

import numpy as np
import pytest

from core import GroundTruth
from evaluation.aligners.nearest_time_index_aligner import (
    MAX_ALLOWED_TIME_DIFFERENCE,
    NearestTimeIndexAligner,
)
from evaluation.core.kalman_estimate import KalmanEstimate


def _make_gt(time: np.ndarray) -> GroundTruth:
    n = len(time)
    ones3 = np.ones((n, 3))
    quat = np.zeros((n, 4))
    quat[:, 0] = 1.0  # identity quaternion [w=1, x=0, y=0, z=0]
    return GroundTruth(
        time=time,
        position=ones3.copy(),
        geo_coordinates=ones3.copy(),
        velocity=np.zeros((n, 3)),
        acceleration=np.zeros((n, 3)),
        attitude=quat,
        angular_velocity=np.zeros((n, 3)),
        pressure=np.ones(n) * 101_325.0,
        simulation_sampling_rate=1.0 / float(np.mean(np.diff(time))) if n > 1 else 100.0,
    )


def _make_estimate(time: np.ndarray, rate: float = 100.0) -> KalmanEstimate:
    n = len(time)
    ones3 = np.ones((n, 3))
    quat = np.zeros((n, 4))
    quat[:, 0] = 1.0
    return KalmanEstimate(
        time=time,
        position=ones3.copy(),
        geo_coordinates=ones3.copy(),
        velocity=np.zeros((n, 3)),
        acceleration=np.zeros((n, 3)),
        attitude=quat,
        angular_velocity=np.zeros((n, 3)),
        pressure=np.ones(n) * 101_325.0,
        estimation_output_data_rate=rate,
    )


ALIGNER = NearestTimeIndexAligner()


class TestNearestIndices:
    def test_exact_match_returns_own_index(self):
        src = np.array([0.0, 1.0, 2.0, 3.0])
        dst = np.array([0.0, 1.0, 2.0, 3.0])
        idx = ALIGNER._nearest_indices(src, dst)
        np.testing.assert_array_equal(idx, [0, 1, 2, 3])

    def test_closer_to_left_picks_left(self):
        src = np.array([0.0, 1.0, 2.0])
        dst = np.array([0.1])  # closer to 0.0 than to 1.0
        idx = ALIGNER._nearest_indices(src, dst)
        assert idx[0] == 0

    def test_closer_to_right_picks_right(self):
        src = np.array([0.0, 1.0, 2.0])
        dst = np.array([0.9])  # closer to 1.0 than to 0.0
        idx = ALIGNER._nearest_indices(src, dst)
        assert idx[0] == 1

    def test_equidistant_picks_right(self):
        # dst - left == right - dst → condition is False → picks right (idx)
        src = np.array([0.0, 1.0, 2.0])
        dst = np.array([0.5])  # exactly between 0.0 and 1.0
        idx = ALIGNER._nearest_indices(src, dst)
        assert idx[0] == 1

    def test_before_first_source_clamped_to_zero(self):
        src = np.array([1.0, 2.0, 3.0])
        dst = np.array([0.0])
        idx = ALIGNER._nearest_indices(src, dst)
        assert idx[0] == 0

    def test_after_last_source_clamped_to_last(self):
        src = np.array([0.0, 1.0, 2.0])
        dst = np.array([5.0])
        idx = ALIGNER._nearest_indices(src, dst)
        assert idx[0] == 2


class TestAlignAlreadyAligned:
    def test_identical_time_returns_inputs_unchanged(self):
        time = np.linspace(0, 10, 101)
        gt = _make_gt(time)
        est = _make_estimate(time)
        out_est, out_gt = ALIGNER.align(est, gt)
        assert out_est is est
        assert out_gt is gt

    def test_allclose_but_not_identical_times_short_circuits(self):
        time = np.linspace(0, 5, 51)
        gt = _make_gt(time)
        est = _make_estimate(time + 1e-15)  # floating-point noise, allclose passes
        out_est, out_gt = ALIGNER.align(est, gt)
        assert out_est is est
        assert out_gt is gt


class TestAlignResample:
    def test_output_lengths_match_estimate(self):
        gt = _make_gt(np.linspace(0, 10, 801))  # 800 Hz
        est = _make_estimate(np.linspace(0, 10, 101), rate=10.0)  # 10 Hz
        out_est, out_gt = ALIGNER.align(est, gt)
        assert len(out_gt.time) == len(out_est.time) == 101

    def test_resampled_gt_times_equal_estimate_times(self):
        gt = _make_gt(np.linspace(0, 5, 5001))
        est = _make_estimate(np.linspace(0, 5, 51), rate=10.0)
        out_est, out_gt = ALIGNER.align(est, gt)
        np.testing.assert_array_equal(out_gt.time, out_est.time)

    def test_all_gt_arrays_resampled_consistently(self):
        n_gt = 801
        gt_time = np.linspace(0, 10, n_gt)
        gt = _make_gt(gt_time)
        # put a recognisable ramp in position so we can verify which rows were picked
        gt.position[:] = gt_time[:, None]

        est_time = np.linspace(0, 10, 11)
        est = _make_estimate(est_time, rate=1.0)
        _, out_gt = ALIGNER.align(est, gt)

        # each position row should match the nearest GT time
        expected_indices = ALIGNER._nearest_indices(gt_time, est_time)
        np.testing.assert_allclose(out_gt.position, gt.position[expected_indices])


class TestAlignTrim:
    def test_estimate_samples_before_gt_are_removed(self):
        gt = _make_gt(np.linspace(2.0, 10.0, 801))
        est_time = np.linspace(0.0, 10.0, 101)  # starts before GT
        est = _make_estimate(est_time, rate=10.0)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            out_est, out_gt = ALIGNER.align(est, gt)
        assert out_est.time[0] >= gt.time[0]
        assert len(out_est.time) == len(out_gt.time)

    def test_estimate_samples_after_gt_are_removed(self):
        gt = _make_gt(np.linspace(0.0, 8.0, 801))
        est_time = np.linspace(0.0, 10.0, 101)  # ends after GT
        est = _make_estimate(est_time, rate=10.0)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            out_est, out_gt = ALIGNER.align(est, gt)
        assert out_est.time[-1] <= gt.time[-1]
        assert len(out_est.time) == len(out_gt.time)

    def test_trim_emits_warning(self):
        gt = _make_gt(np.linspace(2.0, 8.0, 601))
        est = _make_estimate(np.linspace(0.0, 10.0, 101), rate=10.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ALIGNER.align(est, gt)
        messages = [str(w.message) for w in caught]
        assert any("trimmed" in m for m in messages)

    def test_trim_metadata_recorded_on_estimate(self):
        gt = _make_gt(np.linspace(2.0, 8.0, 601))
        est = _make_estimate(np.linspace(0.0, 10.0, 101), rate=10.0)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            out_est, _ = ALIGNER.align(est, gt)
        assert out_est.metadata.get("trimmed_to_gt_range") is True
        assert "original_sample_count" in out_est.metadata

    def test_no_trim_needed_emits_no_warning(self):
        gt = _make_gt(np.linspace(0.0, 10.0, 1001))
        est = _make_estimate(np.linspace(0.0, 10.0, 101), rate=10.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ALIGNER.align(est, gt)
        trim_warnings = [w for w in caught if "trimmed" in str(w.message)]
        assert len(trim_warnings) == 0


class TestCheckTimeDifference:
    def test_large_time_gap_emits_warning(self):
        picked = np.array([0.0, 1.005, 2.010])  # > 1 ms off
        target = np.array([0.0, 1.000, 2.000])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ALIGNER._check_time_difference(picked, target)
        assert any(MAX_ALLOWED_TIME_DIFFERENCE == pytest.approx(0.001) for _ in [1])
        assert any("time difference" in str(w.message).lower() for w in caught)

    def test_small_time_gap_no_warning(self):
        picked = np.array([0.0, 1.0001, 2.0001])  # < 1 ms off
        target = np.array([0.0, 1.0000, 2.0000])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ALIGNER._check_time_difference(picked, target)
        time_warnings = [w for w in caught if "time difference" in str(w.message).lower()]
        assert len(time_warnings) == 0
