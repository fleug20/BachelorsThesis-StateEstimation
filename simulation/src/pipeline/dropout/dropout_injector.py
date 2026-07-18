import numpy as np

from core.sensor_stream import SensorStream, SensorOrigin
from pipeline.dropout.core.dropout_strategy import DropoutStrategy
from pipeline.dropout.dropout_mode import DropoutMode


class DropoutInjector:
    """Applies dropout strategy to a SensorStream.

    Note, if multiple strategies are added, strategies are combined with logical OR.
    Any strategy that marks a sample as dropped causes a drop.

    The DropoutInjector supports different strategies:
    - displaying the result as NaN (np.nan)
    - holding the last valid value (may also contain NaN if no first value existed) and
    - deleting the values entirely. Watch out as this changes the shape of the SensorStream

    A DropoutInjector instance can be applied to all channels (default) or only selected channels.
    """

    def __init__(
            self,
            strategies: list[DropoutStrategy],
            mode: DropoutMode = DropoutMode.NAN,
            seed: int | None = 42,
            channels: list[int] | None = None,
    ):
        if mode is DropoutMode.DELETE and channels is not None:
            raise ValueError(
                "DropoutInjector: DropoutMode.DELETE requires channels=None "
                "(row deletion cannot target a channel subset)."
            )
        self._strategies = list(strategies)
        self._mode = mode
        self._seed = seed
        self._channels = channels

    def apply(self, stream: SensorStream) -> SensorStream:
        out = stream.copy()
        mask = self._build_mask(out.time)
        target_channels = (
            self._channels if self._channels is not None else list(range(out.n_channels))
        )

        # select the right mode
        if self._mode is DropoutMode.DELETE:
            self._apply_delete(out, mask)
        elif self._mode is DropoutMode.NAN:
            self._apply_nan(out, mask, target_channels)
        elif self._mode is DropoutMode.HOLD_LAST:
            self._apply_hold_last(out, mask, target_channels)
        else:
            raise ValueError(f"DropoutStrategy  {self._mode} not implemented...")

        out.origin = SensorOrigin.PIPELINE_PROCESSED
        out.add_processing_history_entry(
            f"DropoutInjector applied mode={self._mode.name} "
            f"seed={self._seed} channels={target_channels} "
            f"strategies=[{', '.join(e.describe() for e in self._strategies)}] "
            f"affected total samples={len(mask[True])}"
        )
        return out

    def _build_mask(self, time: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(time), dtype=bool)
        rng = np.random.default_rng(self._seed)
        for strategy in self._strategies:
            mask |= strategy.generate_mask(time, rng)
        return mask

    @staticmethod
    def _ensure_float(out: SensorStream) -> None:
        if not np.issubdtype(out.data.dtype, np.floating):
            out.data = out.data.astype(np.float64)

    @classmethod
    def _apply_nan(cls, out: SensorStream, mask: np.ndarray, channels: list[int]) -> None:
        if not mask.any():
            return
        cls._ensure_float(out)
        for c in channels:
            out.data[mask, c] = np.nan

    @classmethod
    def _apply_hold_last(cls, out: SensorStream, mask: np.ndarray, channels: list[int]) -> None:
        if not mask.any():
            return
        cls._ensure_float(out)

        n = len(mask)
        idx = np.arange(n)
        idx[mask] = -1
        np.maximum.accumulate(idx, out=idx)

        has_prior = mask & (idx >= 0)
        leading = mask & (idx < 0)

        for c in channels:
            col = out.data[:, c].copy()
            out.data[has_prior, c] = col[idx[has_prior]]
            out.data[leading, c] = np.nan

    @staticmethod
    def _apply_delete(out: SensorStream, mask: np.ndarray) -> None:
        if not mask.any():
            return
        keep = ~mask
        out.time = out.time[keep]
        out.data = out.data[keep, :]
