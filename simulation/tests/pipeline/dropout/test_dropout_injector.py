import numpy as np
import pytest

from core.sensor_stream import SensorOrigin
from pipeline.dropout.core.dropout_strategy import WindowDropout, RandomDropout
from pipeline.dropout.dropout_injector import DropoutInjector
from pipeline.dropout.dropout_mode import DropoutMode

# Window that drops samples at index 2 and 3
DROP_WINDOW = [(0.015, 0.035)]


class TestDropoutInjectorNan:
    def test_dropped_samples_become_nan(self, make_stream):
        stream = make_stream(np.ones(10))
        injector = DropoutInjector([WindowDropout(DROP_WINDOW)], mode=DropoutMode.NAN)
        out = injector.apply(stream)
        assert np.isnan(out.data[2, 0]) and np.isnan(out.data[3, 0])

    def test_non_dropped_samples_unchanged(self, make_stream):
        stream = make_stream(np.ones(10))
        injector = DropoutInjector([WindowDropout(DROP_WINDOW)], mode=DropoutMode.NAN)
        out = injector.apply(stream)
        keep = [i for i in range(10) if i not in (2, 3)]
        np.testing.assert_array_equal(out.data[keep, 0], 1.0)

    def test_channel_selection_only_affects_specified_channels(self, make_multi_stream):
        data = np.ones((10, 3))
        stream = make_multi_stream(data)
        injector = DropoutInjector(
            [WindowDropout(DROP_WINDOW)], mode=DropoutMode.NAN, channels=[0, 2]
        )
        out = injector.apply(stream)
        # channels 0 and 2 have NaN at dropped rows
        assert np.isnan(out.data[2, 0]) and np.isnan(out.data[2, 2])
        # channel 1 is untouched
        assert out.data[2, 1] == 1.0

    def test_no_dropout_leaves_data_unchanged(self, make_stream):
        data = np.arange(10, dtype=float)
        stream = make_stream(data)
        injector = DropoutInjector([RandomDropout(0.0)], mode=DropoutMode.NAN)
        out = injector.apply(stream)
        np.testing.assert_array_equal(out.data[:, 0], data)


class TestDropoutInjectorHoldLast:
    def test_dropped_sample_gets_previous_value(self, make_stream):
        data = np.arange(10, dtype=float)
        stream = make_stream(data)
        # drops indices 2 and 3
        injector = DropoutInjector([WindowDropout(DROP_WINDOW)], mode=DropoutMode.HOLD_LAST)
        out = injector.apply(stream)
        assert out.data[2, 0] == pytest.approx(data[1])
        assert out.data[3, 0] == pytest.approx(data[1])

    def test_leading_dropout_becomes_nan(self, make_stream):
        data = np.arange(10, dtype=float)
        stream = make_stream(data)
        # drop only the very first sample
        injector = DropoutInjector(
            [WindowDropout([(-0.001, 0.001)])], mode=DropoutMode.HOLD_LAST
        )
        out = injector.apply(stream)
        assert np.isnan(out.data[0, 0])
        # remaining samples are intact
        np.testing.assert_array_equal(out.data[1:, 0], data[1:])


class TestDropoutInjectorDelete:
    def test_dropped_rows_are_removed(self, make_stream):
        stream = make_stream(np.ones(10))
        injector = DropoutInjector([WindowDropout(DROP_WINDOW)], mode=DropoutMode.DELETE)
        out = injector.apply(stream)
        assert out.n_samples == 8
        assert out.time.shape == (8,)
        assert out.data.shape == (8, 1)

    def test_delete_with_channel_selection_raises(self, make_stream):
        make_stream(np.ones(10))
        with pytest.raises(ValueError, match="DELETE"):
            DropoutInjector(
                [WindowDropout(DROP_WINDOW)], mode=DropoutMode.DELETE, channels=[0]
            )


class TestDropoutInjectorGeneral:
    def test_output_origin_is_pipeline_processed(self, make_stream):
        stream = make_stream(np.ones(10))
        injector = DropoutInjector([WindowDropout(DROP_WINDOW)])
        out = injector.apply(stream)
        assert out.origin == SensorOrigin.PIPELINE_PROCESSED

    def test_processing_history_is_updated(self, make_stream):
        stream = make_stream(np.ones(10))
        n_before = len(stream.processing_history)
        injector = DropoutInjector([WindowDropout(DROP_WINDOW)])
        out = injector.apply(stream)
        assert len(out.processing_history) == n_before + 1

    def test_apply_does_not_mutate_input(self, make_stream):
        data = np.ones(10)
        stream = make_stream(data.copy())
        original_data = stream.data.copy()
        injector = DropoutInjector([WindowDropout(DROP_WINDOW)])
        injector.apply(stream)
        np.testing.assert_array_equal(stream.data, original_data)

    def test_multiple_strategies_combined(self, make_stream):
        stream = make_stream(np.ones(20))
        # window drops index 2-3, random p=0 drops nothing → result same as window alone
        injector = DropoutInjector(
            [WindowDropout(DROP_WINDOW), RandomDropout(0.0)], mode=DropoutMode.NAN
        )
        out = injector.apply(stream)
        assert np.isnan(out.data[2, 0]) and np.isnan(out.data[3, 0])
        non_dropped = [i for i in range(20) if i not in (2, 3)]
        assert not np.any(np.isnan(out.data[non_dropped, 0]))
