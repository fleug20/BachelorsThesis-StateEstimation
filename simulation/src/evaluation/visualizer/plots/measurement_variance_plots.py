import matplotlib.pyplot as plt
import numpy as np

from evaluation.core import EvaluationResult

_AXIS_LABELS: dict[str, list[str]] = {
    "gnss": ["N", "E", "D"],
    "tof": ["altitude"],
}


def plot_measurement_variances(result: EvaluationResult, as_stddev: bool = True,
                               axes: list[plt.Axes] | None = None, ) -> plt.Figure:
    """One subplot per sensor in measurement_variances, showing std dev or variance over time."""
    mvar = result.estimate.measurement_variances
    if not mvar:
        raise ValueError("No measurement_variances recorded in this estimate.")

    keys = list(mvar.keys())
    n = len(keys)

    if axes is None:
        fig, axs = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)
        axs = [axs[i, 0] for i in range(n)]
    else:
        axs = list(axes)
        fig = axs[0].get_figure()

    t = result.estimate.time
    ylabel = "std dev [m]" if as_stddev else "variance [m²]"
    kind = "std dev" if as_stddev else "variance"

    for ax, key in zip(axs, keys):
        data = np.sqrt(mvar[key]) if as_stddev else mvar[key].copy()  # (N, n_ch)
        n_ch = data.shape[1]
        base_labels = _AXIS_LABELS.get(key, [])
        if len(base_labels) >= n_ch:
            axis_labels = base_labels[:n_ch]
        else:
            prefix = base_labels[0] if base_labels else key
            axis_labels = [f"{prefix} {i}" for i in range(n_ch)]

        for i, label in enumerate(axis_labels):
            ax.plot(t, data[:, i], label=label, linewidth=1)

        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{key.upper()} measurement {kind}")
        if data.shape[1] > 1:
            ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_measurement_variance_comparison(results: list[EvaluationResult], sensor_key: str,
                                         labels: list[str] | None = None, as_stddev: bool = True,
                                         ax: plt.Axes | None = None, ) -> plt.Axes:
    """Overlay measurement std dev / variance for one sensor across multiple variants"""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    labels = labels or [f"run {i}" for i in range(len(results))]
    ylabel = "std dev [m]" if as_stddev else "variance [m²]"
    kind = "std dev" if as_stddev else "variance"

    for result, label in zip(results, labels):
        mvar = result.estimate.measurement_variances
        if sensor_key not in mvar:
            continue
        data = np.sqrt(mvar[sensor_key]) if as_stddev else mvar[sensor_key].copy()  # (N, n_ch)
        t = result.estimate.time
        series = data[:, 0] if data.shape[1] == 1 else np.nanmean(data, axis=1)
        ax.plot(t, series, label=label, linewidth=1)

    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{sensor_key.upper()} measurement {kind} — comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax
