from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class CSVDataLoader:
    """IMPORTANT: If you want to save and load a SimulationResult, please use the builtin .save() and .load() methods of the SimulationResult class.

    This class provides functionality to load data exported by SimulationResult.
    Holds the individual sensors and is able to produce different DataFrame outputs (like a unified df)

    The unified DataFrame uses the fastest sensor's time grid as its index.
    Slower sensors have NaN at timestamps where they did not sample.
    Ground-truth columns are resampled onto the same grid via asof merge.
    https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html
    """

    def __init__(
            self,
            sensors_clean: dict[str, pd.DataFrame],
            sensors_noisy: dict[str, pd.DataFrame],
            ground_truth_df: pd.DataFrame,
            metadata: dict,
    ) -> None:
        self.sensors_clean = sensors_clean
        self.sensors_noisy = sensors_noisy
        self.ground_truth_df = ground_truth_df
        self.metadata = metadata

    @classmethod
    def from_export(cls, export_dir: str | Path) -> CSVDataLoader:
        """Load a SimulationResult from a directory written by SimulationResult.export_data()."""
        path = Path(export_dir)

        with open(path / "metadata.yaml") as f:
            metadata = yaml.safe_load(f)

        # read clean sensors
        clean_names: list[str] = metadata.get("sensors_clean", [])
        clean_dfs: dict[str, pd.DataFrame] = {}
        for name in clean_names:
            print(f"CSVDataLoader: loading sensor {name}...")
            csv_path = path / f"{name}.csv"
            clean_dfs[name] = pd.read_csv(csv_path)

        # read noisy sensors
        noisy_names: list[str] = metadata.get("sensors_noisy", [])
        noisy_dfs: dict[str, pd.DataFrame] = {}
        for name in noisy_names:
            print(f"CSVDataLoader: loading sensor {name}...")
            csv_path = path / f"{name}.csv"
            noisy_dfs[name] = pd.read_csv(csv_path)

        # read ground truth
        ground_truth_df = pd.read_csv((path / "ground_truth.csv"))

        return cls(clean_dfs, noisy_dfs, ground_truth_df, metadata)

    def get_sensors_noisy_dataframe(self) -> pd.DataFrame:
        """Merges all noisy sensor data into a single DataFrame with the fastest sensor's time grid.
        No ground truth data will be added"""
        if not self.sensors_noisy:
            raise ValueError("No noisy sensors found.")

        # find the noisy sensor with most samples (= largest Hz rate)
        fastest_name = max(self.sensors_noisy, key=lambda n: len(self.sensors_noisy[n]))
        fastest_df = self.sensors_noisy[fastest_name]
        print(f"CSVDataLoader: using {fastest_name} as fastest sensor.")

        # tolerance: half the fastest sensor's time step
        dt = np.diff(fastest_df["time"].values[:2])[0]
        tolerance = dt / 2

        # create df from fastest sensor
        merged = fastest_df.rename(
            columns={c: f"{fastest_name}.{c}" for c in fastest_df.columns if c != "time"}
        ).sort_values("time").reset_index(drop=True)

        # merge rest of noisy sensors onto fastest sensor
        for name, sensor_df in self.sensors_noisy.items():
            if name == fastest_name:
                continue
            prefixed = sensor_df.rename(
                columns={c: f"{name}.{c}" for c in sensor_df.columns if c != "time"}
            ).sort_values("time").reset_index(drop=True)
            merged = pd.merge_asof(
                merged, prefixed, on="time", direction="nearest", tolerance=tolerance,
            )

        return merged

    def get_sensors_clean_dataframe(self) -> pd.DataFrame:
        """Merges all clean sensor data into a single DataFrame with the fastest sensor's time grid.
        No ground truth data will be added"""
        if not self.sensors_clean:
            raise ValueError("No clean sensors found.")

        # find the noisy sensor with most samples (= largest Hz rate)
        fastest_name = max(self.sensors_clean, key=lambda n: len(self.sensors_clean[n]))
        fastest_df = self.sensors_clean[fastest_name]
        print(f"CSVDataLoader: using {fastest_name} as fastest sensor.")

        # tolerance: half the fastest sensor's time step
        dt = np.diff(fastest_df["time"].values[:2])[0]
        tolerance = dt / 2

        # create df from fastest sensor
        merged = fastest_df.rename(
            columns={c: f"{fastest_name}.{c}" for c in fastest_df.columns if c != "time"}
        ).sort_values("time").reset_index(drop=True)

        # merge rest of clean sensors onto fastest sensor
        for name, sensor_df in self.sensors_clean.items():
            if name == fastest_name:
                continue
            prefixed = sensor_df.rename(
                columns={c: f"{name}.{c}" for c in sensor_df.columns if c != "time"}
            ).sort_values("time").reset_index(drop=True)
            merged = pd.merge_asof(
                merged, prefixed, on="time", direction="nearest", tolerance=tolerance,
            )

        return merged

    def get_unified_dataframe(self) -> pd.DataFrame:
        """Merges all sensor data into a single DataFrame with the fastest sensor's time grid as index.
        The ground truth will be resampled onto the same grid.
        """
        if not self.sensors_clean and not self.sensors_noisy:
            return self.ground_truth_df.copy()

        # find the sensor with most samples (= largest Hz rate)
        sensor_dfs = {**self.sensors_clean, **self.sensors_noisy}
        fastest_name = max(sensor_dfs, key=lambda n: len(sensor_dfs[n]))
        fastest_df = sensor_dfs[fastest_name]
        print(f"CSVDataLoader: using {fastest_name} as fastest sensor.")

        # tolerance: half the fastest sensor's time step
        dt = np.diff(fastest_df["time"].values[:2])[0]
        tolerance = dt / 2

        # create df from fastest sensor
        unified = fastest_df.rename(
            columns={c: f"{fastest_name}.{c}" for c in fastest_df.columns if c != "time"}
        ).sort_values("time").reset_index(drop=True)

        # merge rest of sensors onto fastest sensor
        for name, df in sensor_dfs.items():
            if name == fastest_name:
                continue
            prefixed = df.rename(
                columns={c: f"{name}.{c}" for c in df.columns if c != "time"}
            ).sort_values("time").reset_index(drop=True)
            unified = pd.merge_asof(
                unified, prefixed, on="time", direction="nearest", tolerance=tolerance,
            )

        # merge ground truth. Watch out that ground truth is dense enough!
        gt = self.ground_truth_df.rename(
            columns={c: f"ground_truth.{c}" for c in self.ground_truth_df.columns if c != "time"}
        ).sort_values("time").reset_index(drop=True)
        unified = pd.merge_asof(unified, gt, on="time", direction="nearest")

        return unified
