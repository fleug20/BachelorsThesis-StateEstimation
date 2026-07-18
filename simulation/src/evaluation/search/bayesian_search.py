from typing import Callable

import optuna

from core import SimulationResult
from evaluation.core import KalmanRunner, Metric, MetricResult, TimeAligner
from evaluation.evaluator import Evaluator
from evaluation.search.search_params import SearchParams
from evaluation.search.search_result import SearchRecord, SearchResult


def _average_metrics(records: list[SearchRecord]) -> dict[str, MetricResult]:
    """Average metric values all evaluated simulation runs (in one trial = one parameter set)"""
    successful = [r for r in records if r.success]
    if not successful:
        return {}
    metric_names = list(successful[0].metrics.keys())
    result = {}
    for name in metric_names:
        template = successful[0].metrics[name]
        avg_value = sum(r.metrics[name].value for r in successful) / len(successful)
        result[name] = MetricResult(
            name=template.name,
            value=avg_value,
            unit=template.unit,
            per_axis=None,
        )
    return result


class BayesianSearch:

    def __init__(
            self,
            runner_factory: Callable[[dict], KalmanRunner],
            search_params: SearchParams,
            metrics: list[Metric],
            objective: Callable[[dict[str, MetricResult]], float | list[float]],
            aligner: TimeAligner | None = None,
            n_trials: int = 100,
            directions: list[str] | str = "minimize",
            study_name: str | None = None,
            use_storage: bool = False,
    ):
        self._runner_factory = runner_factory
        self._search_params = search_params
        self._metrics = metrics
        self._objective = objective
        self._aligner = aligner
        self._n_trials = n_trials
        self._directions = directions
        self._study_name = study_name
        self._use_storage = use_storage

    @property
    def _is_multi_objective(self) -> bool:
        return isinstance(self._directions, list)

    def run(self, simulation_results: list[SimulationResult]) -> SearchResult:
        """Execute Bayesian search across all SimulationResults and return all records."""
        all_records: list[SearchRecord] = []
        n_objectives = len(self._directions) if self._is_multi_objective else 1
        failure_value = tuple(float("inf") for _ in range(n_objectives)) if self._is_multi_objective else float("inf")

        def optuna_objective(trial: optuna.Trial) -> float | tuple[float, ...]:
            params = self._search_params.suggest(trial)
            runner = self._runner_factory(params)
            evaluator = Evaluator(runner, self._metrics, self._aligner)

            trial_records: list[SearchRecord] = []
            for sim in simulation_results:
                name = sim.metadata.get("name", str(id(sim)))
                try:
                    eval_result = evaluator.evaluate(sim)
                    trial_records.append(SearchRecord(
                        params=params,
                        simulation_name=name,
                        metrics=eval_result.metrics,
                        success=True,
                    ))
                except Exception as e:
                    trial_records.append(SearchRecord(
                        params=params,
                        simulation_name=name,
                        metrics={},
                        success=False,
                    ))
                    all_records.extend(trial_records)
                    print(f"Error in simulation {name}: {e}")
                    return failure_value

            all_records.extend(trial_records)
            avg_metrics = _average_metrics(trial_records)
            result = self._objective(avg_metrics)
            return tuple(result) if self._is_multi_objective else result

        # Optuna configuration
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study = optuna.create_study(
            directions=self._directions if self._is_multi_objective else None,
            direction=self._directions if not self._is_multi_objective else None,
            study_name=self._study_name,
            storage="sqlite:///ekf_optimization.db" if self._use_storage else None,
            load_if_exists=False,
        )
        study.optimize(optuna_objective, n_trials=self._n_trials)

        pareto_params = None
        if self._is_multi_objective:
            pareto_params = [t.params for t in study.best_trials]

        return SearchResult(
            records=all_records,
            param_names=list(self._search_params.params.keys()),
            pareto_params=pareto_params,
        )
