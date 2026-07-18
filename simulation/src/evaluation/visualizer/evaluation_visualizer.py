import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from evaluation.core import EvaluationResult
from evaluation.visualizer.plots.error_over_time_plots import (
    plot_position_error,
    plot_altitude_error,
    plot_altitude_error_comparison,
    plot_state_error_norms_comparison,
    plot_velocity_error,
    plot_attitude_error,
)
from evaluation.visualizer.plots.measurement_variance_plots import (
    plot_measurement_variances,
    plot_measurement_variance_comparison,
)
from evaluation.visualizer.plots.nees_over_time_plots import plot_nees_groups_over_time
from evaluation.visualizer.plots.trajectory_plots import plot_multiple_flight_trajectories, plot_trajectory


class EvaluationVisualizer:
    """Produces plots from an EvaluationResult"""

    def __init__(self, result: EvaluationResult):
        self._result = result

    def plot_position_error_over_time(self, ax: plt.Axes | None = None, show_events: bool = True) -> plt.Axes:
        return plot_position_error(self._result, ax=ax, show_events=show_events)

    def plot_altitude_error_over_time(self, ax: plt.Axes | None = None, show_events: bool = True) -> plt.Axes:
        return plot_altitude_error(self._result, ax=ax, show_events=show_events)

    def plot_velocity_error_over_time(self, ax: plt.Axes | None = None, show_events: bool = True) -> plt.Axes:
        return plot_velocity_error(self._result, ax=ax, show_events=show_events)

    def plot_attitude_error_over_time(self, ax: plt.Axes | None = None, show_events: bool = True) -> plt.Axes:
        return plot_attitude_error(self._result, ax=ax, show_events=show_events)

    def plot_trajectory(self, ax: plt.Axes | None = None) -> plt.Axes:
        return plot_trajectory(self._result, ax=ax)

    def plot_state_errors(self, show_events: bool = True) -> plt.Figure:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
        self.plot_position_error_over_time(ax=ax1, show_events=show_events)
        self.plot_attitude_error_over_time(ax=ax2, show_events=show_events)
        self.plot_velocity_error_over_time(ax=ax3, show_events=show_events)
        fig.tight_layout()
        return fig

    def plot_measurement_variances(self, as_stddev: bool = True, axes: list[plt.Axes] | None = None) -> plt.Figure:
        return plot_measurement_variances(self._result, as_stddev=as_stddev, axes=axes)

    @staticmethod
    def plot_measurement_variance_comparison(results: list[EvaluationResult], sensor_key: str,
                                             labels: list[str] | None = None, as_stddev: bool = True,
                                             ax: plt.Axes | None = None, ) -> plt.Axes:
        return plot_measurement_variance_comparison(results, sensor_key, labels=labels, as_stddev=as_stddev, ax=ax)

    def plot_nees_groups_over_time(self, ax: plt.Axes | None = None, ) -> Axes:
        return plot_nees_groups_over_time(self._result, ax=ax)

    @staticmethod
    def plot_state_error_norms_comparison(results: list[EvaluationResult],
                                          labels: list[str] | None = None, ) -> plt.Figure:
        return plot_state_error_norms_comparison(results, labels=labels)

    @staticmethod
    def plot_altitude_error_comparison(results: list[EvaluationResult], labels: list[str] | None = None,
                                       ax: plt.Axes | None = None, ) -> plt.Axes:
        return plot_altitude_error_comparison(results, labels=labels, ax=ax)

    @staticmethod
    def plot_multiple_flight_trajectories(results: list[EvaluationResult], labels: list[str] | None = None,
                                          show_estimate: bool = True, title: str = "Flight trajectories"):
        return plot_multiple_flight_trajectories(results, labels=labels, show_estimate=show_estimate, title=title)
