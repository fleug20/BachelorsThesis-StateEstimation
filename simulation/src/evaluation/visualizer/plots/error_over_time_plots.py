import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory
from scipy.spatial.transform import Rotation

from evaluation.core import EvaluationResult


def _add_event_lines(ax: plt.Axes, result: EvaluationResult) -> None:
    sim_meta = result.metadata.get("simulation_metadata", {})
    events = [
        (sim_meta.get("drogue_inflation_time"), "drogue"),
        (sim_meta.get("main_parachute_inflation_time"), "main chute"),
    ]
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for t, label in events:
        if t is not None:
            ax.axvline(t, color="black", linestyle="--", linewidth=0.8)
            ax.text(t, 0.9, label, rotation=90, transform=transform, va="top", ha="left", fontsize=9)


def plot_position_error(result: EvaluationResult, ax: plt.Axes | None = None, show_events: bool = True, ) -> plt.Axes:
    """Per-axis position error (estimate - ground truth) over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    t = result.estimate.time
    diff = result.estimate.position - result.ground_truth.position  # (N, 3)

    ax.plot(t, diff[:, 0], label="x", linewidth=1)
    ax.plot(t, diff[:, 1], label="y", linewidth=1)
    ax.plot(t, diff[:, 2], label="z", linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    if show_events:
        _add_event_lines(ax, result)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Position error (estimate - ground truth)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_altitude_error(result: EvaluationResult, ax: plt.Axes | None = None, show_events: bool = True, ) -> plt.Axes:
    """Altitude error (estimate - ground truth) over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    t = result.estimate.time
    diff = result.estimate.position[:, 2] - result.ground_truth.position[:, 2]  # (N)

    ax.plot(t, diff, label="altitude", linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    if show_events:
        _add_event_lines(ax, result)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("altitude error [m]")
    ax.set_title("Altitude error (estimate - ground truth)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_state_error_norms_comparison(results: list[EvaluationResult], labels: list[str] | None = None, ) -> plt.Figure:
    labels = labels or [f"run {i}" for i in range(len(results))]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))

    _plot_norm_series(results, labels, ax1, _pos_norm, "position error [m]", "Position error norm")
    _plot_norm_series(results, labels, ax2, _vel_norm, "velocity error [m/s]", "Velocity error norm")
    _plot_norm_series(results, labels, ax3, _att_norm, "attitude error [deg]", "Attitude error norm")

    fig.tight_layout()
    return fig


def _pos_norm(result: EvaluationResult) -> tuple[np.ndarray, np.ndarray]:
    t = result.estimate.time
    return t, np.linalg.norm(result.estimate.position - result.ground_truth.position, axis=1)


def _vel_norm(result: EvaluationResult) -> tuple[np.ndarray, np.ndarray]:
    t = result.estimate.time
    return t, np.linalg.norm(result.estimate.velocity - result.ground_truth.velocity, axis=1)


def _att_norm(result: EvaluationResult) -> tuple[np.ndarray, np.ndarray]:
    t = result.estimate.time
    r_gt = Rotation.from_quat(result.ground_truth.attitude, scalar_first=True)
    r_est = Rotation.from_quat(result.estimate.attitude, scalar_first=True)
    return t, np.degrees(np.linalg.norm((r_gt * r_est.inv()).as_rotvec(), axis=1))


def _plot_norm_series(results: list[EvaluationResult], labels: list[str], ax: plt.Axes, norm_fn, ylabel: str,
                      title: str, ) -> None:
    for result, label in zip(results, labels):
        t, norm = norm_fn(result)
        ax.plot(t, norm, label=label, linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def plot_altitude_error_comparison(results: list[EvaluationResult], labels: list[str] | None = None,
                                   ax: plt.Axes | None = None, ) -> plt.Axes:
    """Altitude error over time for multiple evaluations on the same axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    labels = labels or [f"run {i}" for i in range(len(results))]

    for result, label in zip(results, labels):
        t = result.estimate.time
        diff = result.estimate.position[:, 2] - result.ground_truth.position[:, 2]
        ax.plot(t, diff, label=label, linewidth=1)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("altitude error [m]")
    ax.set_title("Altitude error (estimate - ground truth)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_velocity_error(result: EvaluationResult, ax: plt.Axes | None = None, show_events: bool = True, ) -> plt.Axes:
    """Per-axis velocity error (estimate - ground truth) over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    t = result.estimate.time
    diff = result.estimate.velocity - result.ground_truth.velocity  # (N, 3)

    ax.plot(t, diff[:, 0], label="x", linewidth=1)
    ax.plot(t, diff[:, 1], label="y", linewidth=1)
    ax.plot(t, diff[:, 2], label="z", linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    if show_events:
        _add_event_lines(ax, result)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity error [m/s]")
    ax.set_title("Velocity error (estimate - ground truth)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_attitude_error(result: EvaluationResult, ax: plt.Axes | None = None, show_events: bool = True, ) -> plt.Axes:
    """Per-axis attitude error over time, expressed as rotation-vector components [deg]."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    t = result.estimate.time

    r_gt = Rotation.from_quat(result.ground_truth.attitude, scalar_first=True)
    r_est = Rotation.from_quat(result.estimate.attitude, scalar_first=True)
    r_err = r_gt * r_est.inv()

    err_deg = np.degrees(r_err.as_rotvec())  # (N, 3)

    ax.plot(t, err_deg[:, 0], label="x", linewidth=1)
    ax.plot(t, err_deg[:, 1], label="y", linewidth=1)
    ax.plot(t, err_deg[:, 2], label="z", linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    if show_events:
        _add_event_lines(ax, result)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("attitude error [deg]")
    ax.set_title("Attitude error (estimate - ground truth)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax
