import warnings

import numpy as np

from core import GroundTruth
from evaluation.core import TimeAligner
from evaluation.core.kalman_estimate import KalmanEstimate

MAX_ALLOWED_TIME_DIFFERENCE = 0.001  # 1ms


class NearestTimeIndexAligner(TimeAligner):
    """Resamples a GroundTruth onto a KalmanEstimate time grid"""

    def align(self, estimate: KalmanEstimate, ground_truth: GroundTruth, ) -> tuple[KalmanEstimate, GroundTruth]:
        if estimate.time.shape == ground_truth.time.shape and np.allclose(estimate.time, ground_truth.time):
            return estimate, ground_truth

        # trim the estimate to GT's time range
        t_min = float(ground_truth.time[0])
        t_max = float(ground_truth.time[-1])
        in_range = (estimate.time >= t_min) & (estimate.time <= t_max)
        dropped = int((~in_range).sum())

        if dropped > 0:
            warnings.warn(
                f"NearestTimeIndexAligner: trimmed {dropped} estimate samples outside "
                f"ground-truth range [{t_min:.4f}s, {t_max:.4f}s]"
            )
            target_time = estimate.time[in_range]
            trimmed_estimate = self._slice_estimate(estimate, in_range)
        else:
            target_time = estimate.time
            trimmed_estimate = estimate

        aligned_gt = self._resample_ground_truth(
            ground_truth, target_time, estimate.estimation_output_data_rate
        )

        return trimmed_estimate, aligned_gt

    def describe(self) -> str:
        return f"NearestTimeIndexAligner(max_allowed_time_difference={MAX_ALLOWED_TIME_DIFFERENCE}s)"

    def _resample_ground_truth(self, gt: GroundTruth, target_time: np.ndarray, target_rate: float, ) -> GroundTruth:
        idx = self._nearest_indices(gt.time, target_time)
        self._check_time_difference(gt.time[idx], target_time)
        return GroundTruth(
            time=target_time.copy(),
            position=gt.position[idx],
            geo_coordinates=gt.geo_coordinates[idx],
            velocity=gt.velocity[idx],
            acceleration=gt.acceleration[idx],
            attitude=gt.attitude[idx],
            angular_velocity=gt.angular_velocity[idx],
            pressure=gt.pressure[idx],
            simulation_sampling_rate=target_rate,
            metadata={
                **gt.metadata,
            },
        )

    @staticmethod
    def _slice_estimate(est: KalmanEstimate, mask: np.ndarray) -> KalmanEstimate:
        return KalmanEstimate(
            time=est.time[mask],
            position=est.position[mask],
            geo_coordinates=est.geo_coordinates[mask],
            velocity=est.velocity[mask],
            acceleration=est.acceleration[mask],
            attitude=est.attitude[mask],
            angular_velocity=est.angular_velocity[mask],
            pressure=est.pressure[mask],
            estimation_output_data_rate=est.estimation_output_data_rate,
            covariance=est.covariance[mask] if est.covariance is not None else None,
            metadata={
                **est.metadata,
                "trimmed_to_gt_range": True,
                "original_sample_count": int(len(est.time)),
                "trimmed_sample_count": int(mask.sum()),
            },
        )

    @staticmethod
    def _nearest_indices(src_time: np.ndarray, dst_time: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(src_time, dst_time)
        idx = np.clip(idx, 1, len(src_time) - 1)
        left = src_time[idx - 1]
        right = src_time[idx]
        return np.where(dst_time - left < right - dst_time, idx - 1, idx)

    @staticmethod
    def _check_time_difference(picked_gt_time: np.ndarray, target_time: np.ndarray):
        time_differences = np.abs(picked_gt_time - target_time)
        print(
            f"NearestTimeIndexAligner: total time difference between aligned gt and estimate: "
            f"{time_differences.sum():.6f}s, per sample average: {time_differences.mean():.6f}s"
        )

        if time_differences.mean() > MAX_ALLOWED_TIME_DIFFERENCE:
            warnings.warn(
                f"NearestTimeIndexAligner: average time difference between ground truth and estimate is {time_differences.mean():.6f}s"
                f"(max allowed is {MAX_ALLOWED_TIME_DIFFERENCE}) this may lead to problems during evaluation."
            )
