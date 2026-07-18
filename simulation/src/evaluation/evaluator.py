from datetime import datetime

from core import SimulationResult
from evaluation.core.evaluation_result import EvaluationResult
from evaluation.core.kalman_runner import KalmanRunner
from evaluation.core.metric import Metric, MetricResult
from evaluation.core.time_aligner import TimeAligner


class Evaluator:
    """Orchestrates one evaluation run: drive runner -> align -> calculate metrics -> produce result

    NOTE: if you do not provide a TimeAligner, estimate and ground truth will need to have the same dimensions
    and will be evaluated against each other via index!
    """

    def __init__(self, runner: KalmanRunner, metrics: list[Metric], aligner: TimeAligner | None = None, ):
        self._runner = runner
        self._metrics = list(metrics)
        self._aligner = aligner

    def evaluate(self, simulation_result: SimulationResult) -> EvaluationResult:
        # 1. run the filter
        estimate = self._runner.run(simulation_result)

        # 2. align estimate and ground truth (if TimeAligner is provided)
        gt = simulation_result.ground_truth
        if self._aligner is not None:
            estimate, gt = self._aligner.align(estimate, gt)

        # 3. compute metrics
        metric_results: dict[str, MetricResult] = {}
        for metric in self._metrics:
            result = metric.compute(gt, estimate)
            metric_results[result.name] = result

        # 4. create metadata
        metadata = {
            "evaluated_at": datetime.now().isoformat(),
            "runner": self._runner.describe(),
            "metrics": [m.describe() for m in self._metrics],
            "aligner": self._aligner.describe() if self._aligner is not None else None,
            "simulation_metadata": dict(simulation_result.metadata),
        }

        return EvaluationResult(
            ground_truth=gt,
            estimate=estimate,
            metrics=metric_results,
            metadata=metadata,
        )
