import matplotlib.pyplot as plt
import numpy as np
import plotly.colors
import plotly.graph_objects as go

from evaluation.core import EvaluationResult


def plot_trajectory(result: EvaluationResult, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    gt = result.ground_truth.position  # (N, 3) ENU
    est = result.estimate.position  # (N, 3) ENU

    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="green", linewidth=1.5, label="Ground Truth")
    ax.scatter(gt[0, 0], gt[0, 1], gt[0, 2], color="green", marker="o", s=40, zorder=5)

    ax.plot(est[:, 0], est[:, 1], est[:, 2], color="red", linewidth=1.5, label="Estimate", alpha=0.8)
    ax.scatter(est[0, 0], est[0, 1], est[0, 2], color="red", marker="o", s=40, zorder=5)

    # equal x/y scale so horizontal motion is not distorted
    all_xy = np.concatenate([gt[:, :2], est[:, :2]])
    xy_mid = (all_xy.max(axis=0) + all_xy.min(axis=0)) / 2
    half = np.ptp(all_xy, axis=0).max() / 2
    ax.set_xlim(xy_mid[0] - half, xy_mid[0] + half)
    ax.set_ylim(xy_mid[1] - half, xy_mid[1] + half)

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.set_title("Flight Trajectory")
    ax.legend(loc="best")
    return ax


def plot_multiple_flight_trajectories(results: list[EvaluationResult], labels: list[str] | None = None,
                                      show_estimate: bool = True, title: str = "Flight trajectories", ) -> go.Figure:
    labels = labels or [f"run {i}" for i in range(len(results))]
    colors = plotly.colors.qualitative.Plotly
    fig = go.Figure()

    for i, (result, name) in enumerate(zip(results, labels)):
        color = colors[i % len(colors)]
        gt_pos = result.ground_truth.position  # (N, 3) ENU
        fig.add_trace(go.Scatter3d(
            x=gt_pos[:, 0], y=gt_pos[:, 1], z=gt_pos[:, 2],
            mode="lines", name=f"{name} (GT)",
            line=dict(width=3, color=color),
            legendgroup=name,
        ))
        if show_estimate:
            est_pos = result.estimate.position  # (N, 3) ENU
            fig.add_trace(go.Scatter3d(
                x=est_pos[:, 0], y=est_pos[:, 1], z=est_pos[:, 2],
                mode="lines", name=f"{name} (estimate)",
                line=dict(width=2, color=color, dash="dash"),
                legendgroup=name,
            ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="East [m]",
            yaxis_title="North [m]",
            zaxis_title="Up [m]",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    return fig
