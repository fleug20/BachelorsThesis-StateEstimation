from core.sensor_stream import SensorStream, SensorOrigin
from pipeline.noise.core.noise_profile import NoiseProfile
from pipeline.noise.core.sensor_model import SensorModel


class NoiseInjector:
    """Applies realistic sensor noise to a SensorStream."""

    def __init__(self, profile: NoiseProfile, seed: int | None = None, channels: list[int] | None = None):
        """
        Args:
            profile: Noise profile to apply.
            seed: Random seed for reproducibility. If left empty, numbers are generated randomly.
            channels: List of channel indices to apply noise to. If None, all channels are affected.
        """
        self._profile = profile
        self._seed = seed
        self._channels = channels

    def apply(self, stream: SensorStream) -> SensorStream:
        out = stream.copy()
        dt = out.dt

        target_channels = self._channels if self._channels is not None else list(range(out.n_channels))
        for i in target_channels:
            # use different model per channel to set individual seed. Otherwise, all channels would get the same amount of noise
            model = SensorModel(self._profile, seed=self._seed + i if self._seed is not None else None)
            model.apply(out.data[:, i], dt)

        out.origin = SensorOrigin.PIPELINE_PROCESSED
        out.add_processing_history_entry(
            f"NoiseInjector applied profile '{self._profile.name}' with seed {self._seed} on channels {target_channels}"
        )
        return out
