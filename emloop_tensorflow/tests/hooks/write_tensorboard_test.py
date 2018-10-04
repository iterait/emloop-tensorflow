"""
Test module for :py:mod:`emloop_tensorflow.hooks.write_tensorboard` module.
"""
import logging

import pytest
import numpy as np

from emloop_tensorflow.hooks import WriteTensorBoard

from ..model_test import TrainableModel, _OPTIMIZER


def get_model():
    return TrainableModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output', 'loss'],
                          optimizer=_OPTIMIZER)


def test_args(tmpdir):
    """Test WriteTensorBoard argument handling and ``SummaryWriter`` creation."""
    tmpdir = str(tmpdir)
    model = get_model()

    with pytest.raises(AssertionError):
        _ = WriteTensorBoard(output_dir=tmpdir, model=42)
    with pytest.raises(AssertionError):
        _ = WriteTensorBoard(output_dir=tmpdir, model=model, on_missing_variable='not-recognized')
    with pytest.raises(AssertionError):
        _ = WriteTensorBoard(output_dir=tmpdir, model=model, on_unknown_type='not-recognized')


def test_mock_args(mock_to_writer, tmpdir):
    tmpdir = str(tmpdir)
    model = get_model()

    _ = WriteTensorBoard(output_dir=tmpdir, model=model, flush_secs=42, visualize_graph=True)
    mock_to_writer.assert_called_with(logdir=tmpdir, flush_secs=42, graph=model.graph)

    _ = WriteTensorBoard(output_dir=tmpdir, model=model)
    mock_to_writer.assert_called_with(logdir=tmpdir, flush_secs=10, graph=None)


def test_write(tmpdir, mock_object_writer):
    """Test if ``WriteTensorBoard`` writes to its FileWriter."""
    tmpdir = str(tmpdir)
    hook = WriteTensorBoard(output_dir=tmpdir, model=get_model())

    hook.after_epoch(42, {})
    assert mock_object_writer.call_count == 1
    hook.after_epoch(43, {'valid': {'accuracy': 1.0}})
    assert mock_object_writer.call_count == 2

    hook.after_epoch(44, {'valid': {'accuracy': {'mean': np.float32(1.0)}}})
    hook.after_epoch(45, {'valid': {'accuracy': {'nanmean': 1.0}}})
    hook._summary_writer.close()


def test_image_variable(tmpdir, mock_dict_writer):
    """Test if ``WriteTensorBoard`` checks the image variables properly."""
    tmpdir = str(tmpdir)
    hook = WriteTensorBoard(output_dir=tmpdir, model=get_model(), image_variables=['plot'])

    with pytest.raises(AssertionError):
        hook.after_epoch(0, {'train': {'plot': [None]}})

    with pytest.raises(AssertionError):
        hook.after_epoch(1, {'train': {'plot': np.zeros((10,))}})

    hook.after_epoch(2, {'train': {'plot': np.zeros((10, 10, 3), dtype=np.float32)}})
    hook.after_epoch(2, {'train': {'plot': np.arange(10*10*3).reshape((10, 10, 3)).astype(np.float32)}})
    hook._summary_writer.close()


def test_unknown_type(tmpdir, mock_dict_writer, caplog):
    """Test if ``WriteTensorBoard`` handles unknown variable types as expected."""
    tmpdir = str(tmpdir)
    bad_epoch_data = {'valid': {'accuracy': 'bad_type'}}

    # test ignore
    hook = WriteTensorBoard(output_dir=tmpdir, model=get_model())
    caplog.clear()
    caplog.set_level(logging.INFO)
    hook.after_epoch(42, bad_epoch_data)
    assert caplog.record_tuples == []

    # test warn
    warn_hook = WriteTensorBoard(output_dir=tmpdir, model=get_model(), on_unknown_type='warn')
    caplog.clear()
    caplog.set_level(logging.INFO)
    warn_hook.after_epoch(42, bad_epoch_data)
    assert caplog.record_tuples == [
        ('root', logging.WARNING, 'Variable `accuracy` in stream `valid` has to be of type `int` '
                                  'or `float` (or a `dict` with a key named `mean` or `nanmean` '
                                  'whose corresponding value is of type `int` or `float`), '
                                  'found `<class \'str\'>` instead.')
    ]

    # test error
    raise_hook = WriteTensorBoard(output_dir=tmpdir, model=get_model(), on_unknown_type='error')
    with pytest.raises(ValueError):
        raise_hook.after_epoch(42, bad_epoch_data)

    # test skip image variables
    skip_hook = WriteTensorBoard(output_dir=tmpdir, model=get_model(), on_unknown_type='error',
                                 image_variables=['accuracy'])
    skip_hook.after_epoch(42, {'valid': {'accuracy': np.zeros((10, 10, 3))}})
    skip_hook._summary_writer.close()


def test_missing_variable(tmpdir, mock_dict_writer, caplog):
    """Test if ``WriteTensorBoard`` handles missing image variables as expected."""
    tmpdir = str(tmpdir)
    bad_epoch_data = {'valid': {}}

    # test ignore
    hook = WriteTensorBoard(output_dir=tmpdir, model=get_model(), image_variables=['plot'],
                            on_missing_variable='ignore')
    caplog.clear()
    caplog.set_level(logging.INFO)
    hook.after_epoch(42, bad_epoch_data)
    assert caplog.record_tuples == []

    # test warn
    warn_hook = WriteTensorBoard(output_dir=tmpdir, model=get_model(), image_variables=['plot'],
                                 on_missing_variable='warn')
    caplog.clear()
    caplog.set_level(logging.INFO)
    warn_hook.after_epoch(42, bad_epoch_data)
    assert caplog.record_tuples == [
        ('root', logging.WARNING, '`plot` not found in epoch data.')
    ]

    # test error
    raise_hook = WriteTensorBoard(output_dir=tmpdir, model=get_model(), image_variables=['plot'],
                                  on_missing_variable='error')
    with pytest.raises(KeyError):
        raise_hook.after_epoch(42, bad_epoch_data)
