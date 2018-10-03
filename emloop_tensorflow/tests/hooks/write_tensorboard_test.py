"""
Test module for :py:mod:`emloop_tensorflow.hooks.write_tensorboard` module.
"""
import logging
import unittest.mock as mock

import sys
import numpy as np
import tensorflow as tf
from testfixtures import LogCapture
from emloop.tests.test_core import CXTestCaseWithDir
from emloop_tensorflow.hooks import WriteTensorBoard

from . import cv2_mock
from ..model_test import TrainableModel, _OPTIMIZER


class WriteTensorBoardTest(CXTestCaseWithDir):
    """Test case for :py:class:emloop_tensorflow.hooks.WriteTensorBoard` hook. """

    def get_model(self):
        return TrainableModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output', 'loss'],
                              optimizer=_OPTIMIZER)

    def test_args(self):
        """Test WriteTensorBoard argument handling and ``SummaryWriter`` creation."""
        model = self.get_model()

        with self.assertRaises(AssertionError):
            _ = WriteTensorBoard(output_dir=self.tmpdir, model=42)
        with self.assertRaises(AssertionError):
            _ = WriteTensorBoard(output_dir=self.tmpdir, model=model, on_missing_variable='not-recognized')
        with self.assertRaises(AssertionError):
            _ = WriteTensorBoard(output_dir=self.tmpdir, model=model, on_unknown_type='not-recognized')

        with mock.patch('tensorflow.summary.FileWriter', autospec=True) as mocked_writer:
            _ = WriteTensorBoard(output_dir=self.tmpdir, model=model, flush_secs=42, visualize_graph=True)
            mocked_writer.assert_called_with(logdir=self.tmpdir, flush_secs=42, graph=model.graph)

            _ = WriteTensorBoard(output_dir=self.tmpdir, model=model)
            mocked_writer.assert_called_with(logdir=self.tmpdir, flush_secs=10, graph=None)

    def test_write(self):
        """Test if ``WriteTensorBoard`` writes to its FileWriter."""
        hook = WriteTensorBoard(output_dir=self.tmpdir, model=self.get_model())
        with mock.patch.object(tf.summary.FileWriter, 'add_summary') as mocked_add_summary:
            hook.after_epoch(42, {})
            self.assertEqual(mocked_add_summary.call_count, 1)
            hook.after_epoch(43, {'valid': {'accuracy': 1.0}})
            self.assertEqual(mocked_add_summary.call_count, 2)
        hook.after_epoch(44, {'valid': {'accuracy': {'mean': np.float32(1.0)}}})
        hook.after_epoch(45, {'valid': {'accuracy': {'nanmean': 1.0}}})
        hook._summary_writer.close()

    def test_image_variable(self):
        """Test if ``WriteTensorBoard`` checks the image variables properly."""
        hook = WriteTensorBoard(output_dir=self.tmpdir, model=self.get_model(), image_variables=['plot'])

        with mock.patch.dict('sys.modules', **{'cv2': cv2_mock}):
            with self.assertRaises(AssertionError):
                hook.after_epoch(0, {'train': {'plot': [None]}})

            with self.assertRaises(AssertionError):
                hook.after_epoch(1, {'train': {'plot': np.zeros((10,))}})

            hook.after_epoch(2, {'train': {'plot': np.zeros((10, 10, 3), dtype=np.float32)}})
            hook.after_epoch(2, {'train': {'plot': np.arange(10*10*3).reshape((10, 10, 3)).astype(np.float32)}})
        hook._summary_writer.close()

    def test_unknown_type(self):
        """Test if ``WriteTensorBoard`` handles unknown variable types as expected."""
        bad_epoch_data = {'valid': {'accuracy': 'bad_type'}}

        # test ignore
        hook = WriteTensorBoard(output_dir=self.tmpdir, model=self.get_model())
        with LogCapture(level=logging.INFO) as log_capture:
            hook.after_epoch(42, bad_epoch_data)
        log_capture.check()

        # test warn
        warn_hook = WriteTensorBoard(output_dir=self.tmpdir, model=self.get_model(), on_unknown_type='warn')
        with LogCapture(level=logging.INFO) as log_capture2:
            warn_hook.after_epoch(42, bad_epoch_data)
        log_capture2.check(('root', 'WARNING', 'Variable `accuracy` in stream `valid` has to be of type `int` '
                                               'or `float` (or a `dict` with a key named `mean` or `nanmean` '
                                               'whose corresponding value is of type `int` or `float`), '
                                               'found `<class \'str\'>` instead.'))

        # test error
        raise_hook = WriteTensorBoard(output_dir=self.tmpdir, model=self.get_model(), on_unknown_type='error')
        with self.assertRaises(ValueError):
            raise_hook.after_epoch(42, bad_epoch_data)

        with mock.patch.dict('sys.modules', **{'cv2': cv2_mock}):
            # test skip image variables
            skip_hook = WriteTensorBoard(output_dir=self.tmpdir, model=self.get_model(), on_unknown_type='error',
                                         image_variables=['accuracy'])
            skip_hook.after_epoch(42, {'valid': {'accuracy': np.zeros((10, 10, 3))}})
            skip_hook._summary_writer.close()

    def test_missing_variable(self):
        """Test if ``WriteTensorBoard`` handles missing image variables as expected."""
        bad_epoch_data = {'valid': {}}

        with mock.patch.dict('sys.modules', **{'cv2': cv2_mock}):
            # test ignore
            hook = WriteTensorBoard(output_dir=self.tmpdir, model=self.get_model(), image_variables=['plot'],
                                    on_missing_variable='ignore')
            with LogCapture(level=logging.INFO) as log_capture:
                hook.after_epoch(42, bad_epoch_data)
            log_capture.check()

            # test warn
            warn_hook = WriteTensorBoard(output_dir=self.tmpdir, model=self.get_model(), image_variables=['plot'],
                                         on_missing_variable='warn')
            with LogCapture(level=logging.INFO) as log_capture2:
                warn_hook.after_epoch(42, bad_epoch_data)
            log_capture2.check(('root', 'WARNING', '`plot` not found in epoch data.'))

            # test error
            raise_hook = WriteTensorBoard(output_dir=self.tmpdir, model=self.get_model(), image_variables=['plot'],
                                          on_missing_variable='error')
            with self.assertRaises(KeyError):
                raise_hook.after_epoch(42, bad_epoch_data)
