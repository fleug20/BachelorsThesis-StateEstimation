from dataclasses import dataclass

import pandas as pd

from evaluation.core import MetricResult


@dataclass
class SearchRecord:
    """Result for one simulation run"""
    params: dict
    simulation_name: str
    metrics: dict[str, MetricResult]
    success: bool


@dataclass
class SearchResult:
    """All records collected across a complete search run (multiple simulation runs)"""
    records: list[SearchRecord]
    param_names: list[str]
    pareto_params: list[dict] | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """One row per (params, simulation). Columns: param values + metric values."""
        rows = []
        for rec in self.records:
            row = dict(rec.params)
            row["simulation"] = rec.simulation_name
            row["success"] = rec.success
            for metric_name, metric_result in rec.metrics.items():
                row[metric_name] = metric_result.value
            rows.append(row)
        return pd.DataFrame(rows)

    def aggregate(self) -> pd.DataFrame:
        """Group by param set, compute mean and std of each metric across simulations."""
        df = self.to_dataframe()
        metric_cols = [c for c in df.columns if c not in self.param_names + ["simulation", "success"]]
        agg = df.groupby(self.param_names)[metric_cols].agg(["mean", "std"])
        agg.columns = ["_".join(c) for c in agg.columns]
        return agg.reset_index()

    def best_params(self, metric_name: str, minimize: bool = True) -> dict:
        """Return the parameter set with the best mean value for the given metric.

        NOTE: For multi-objective runs use pareto_dataframe() instead.
        """
        if self.pareto_params is not None:
            raise ValueError(
                "This is a multi-objective result. Use .pareto_dataframe() to inspect the results."
            )
        agg = self.aggregate()
        col = f"{metric_name}_mean"
        if col not in agg.columns:
            raise ValueError(f"Metric '{metric_name}' not found. Available: {list(agg.columns)}")
        idx = agg[col].idxmin() if minimize else agg[col].idxmax()
        return {p: agg.loc[idx, p] for p in self.param_names}

    def pareto_dataframe(self) -> pd.DataFrame:
        if self.pareto_params is None:
            raise ValueError(
                "This is a single-objective result. Use .best_params() instead."
            )
        agg = self.aggregate()
        pareto_df = pd.DataFrame(self.pareto_params)
        return pareto_df.merge(agg, on=self.param_names, how="left").reset_index(drop=True)
