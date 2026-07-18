from enum import Enum, auto


class DropoutMode(Enum):
    """How a dropped sample is represented in the output SensorStream."""

    NAN = auto()
    """Fill dropped samples with np.nan (default)."""

    HOLD_LAST = auto()
    """Replace dropped samples with the most recent valid value (stale data).
    Should simulate a "stuck" sensor"""

    DELETE = auto()
    """Remove dropped rows entirely from the SensorStream.
    Watch out as dimensions of the sensor change and may not match
    up with the ground truth version of the sensor anymore!"""
