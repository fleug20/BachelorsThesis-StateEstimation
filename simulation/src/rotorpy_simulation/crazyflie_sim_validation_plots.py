import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

from core.simulation_result import SimulationResult
from pipeline.synthetic import SyntheticUWBTransceiver

_TOF_KEY = "time_of_flight_distance_clean"


def _sensor_colors(n: int) -> list:
    cmap = matplotlib.colormaps['tab10' if n <= 10 else 'tab20']
    return [cmap(i % cmap.N) for i in range(n)]


def _get_uwb_sensors(synthetic_sensors: list) -> list:
    return [s for s, _, _ in synthetic_sensors if isinstance(s, SyntheticUWBTransceiver)]


def _check_sensors(result: SimulationResult, synthetic_sensors: list) -> list:
    missing = []
    if _TOF_KEY not in result.sensors_clean:
        missing.append(_TOF_KEY)
    uwb_sensors = _get_uwb_sensors(synthetic_sensors)
    if not uwb_sensors:
        missing.append("at least one SyntheticUWBTransceiver in synthetic_sensors")
    for s in uwb_sensors:
        key = f"{s.name}_clean"
        if key not in result.sensors_clean:
            missing.append(key)
    if missing:
        raise ValueError(
            "plot_crazyflie: required sensors not found in result:\n  - "
            + "\n  - ".join(missing)
            + "\nRun the synthetic sensor generation cell first."
        )
    return uwb_sensors


def plot_trajectory_3d(result: SimulationResult, synthetic_sensors: list) -> None:
    """3-D trajectory with UWB anchors, ToF altitude lines, and anchor-to-drone distance lines."""
    uwb_sensors = _check_sensors(result, synthetic_sensors)

    gt = result.ground_truth
    tof_df = result.sensors_clean[_TOF_KEY].to_dataframe()
    drone_pos_df = gt.to_dataframe()[["time", "x", "y", "z"]]
    tof_df = pd.merge_asof(tof_df, drone_pos_df, on="time")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt.position[:, 0], gt.position[:, 1], gt.position[:, 2],
            color="royalblue", linewidth=1, label="Drone trajectory")
    ax.scatter(*gt.position[0], color="green", s=80, marker="o", zorder=5, label="Start")
    ax.scatter(*gt.position[-1], color="red", s=80, marker="x", zorder=5, label="End")

    step = 50
    for i in range(0, len(tof_df), step):
        x, y, alt = tof_df["x"].iloc[i], tof_df["y"].iloc[i], tof_df["altitude"].iloc[i]
        label = "ToF altitude" if i == 0 else None
        ax.plot([x, x], [y, y], [0, alt], color="gray", linewidth=0.5, alpha=0.4, label=label)

    sample_idx = len(gt.time) // 2
    drone_mid = gt.position[sample_idx]
    ax.scatter(*drone_mid, color="black", s=100, marker="D", zorder=6,
               label=f"Drone @ t={gt.time[sample_idx]:.1f} s")

    colors = _sensor_colors(len(uwb_sensors))
    for sensor, color in zip(uwb_sensors, colors):
        ap = sensor.anchor_position
        ax.scatter(*ap, color=color, s=120, marker="^", zorder=5,
                   label=f"{sensor.name} (id={sensor.anchor_id})")
        ax.plot([ap[0], drone_mid[0]], [ap[1], drone_mid[1]], [ap[2], drone_mid[2]],
                color=color, linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Drone Trajectory & UWB Anchor Positions")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_uwb_distances(result: SimulationResult, synthetic_sensors: list) -> None:
    """2-D plot of UWB anchor distances to the drone over time."""
    uwb_sensors = _check_sensors(result, synthetic_sensors)

    colors = _sensor_colors(len(uwb_sensors))
    fig, ax = plt.subplots(figsize=(12, 5))
    for sensor, color in zip(uwb_sensors, colors):
        df = result.sensors_clean[f"{sensor.name}_clean"].to_dataframe()
        ax.plot(df["time"], df["distance"], color=color, linewidth=1, label=sensor.name)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("UWB Anchor Distances to Drone over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
