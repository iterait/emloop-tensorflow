"""
Test module for the rnn models implementation.
"""
import tensorflow as tf

from unittest import TestCase
from cxflow_tensorflow.models.blocks import UnrecognizedCodeError
from cxflow_tensorflow.models.rnn_blocks import RNNBlock

_VALID_CODES = {RNNBlock: ['RNN23', 'LSTM32', 'GRU89', 'biRNN23', 'biLSTM32', 'biGRU89']}

_INVALID_CODES = {RNNBlock: ['32c3', 'rnn32', 'bi-GRU', 'LSTM', '32RNN']}


class RNNBlocksTest(TestCase):
    """Test rnn blocks parsing and applying."""

    def test_valid(self):
        """Test if valid codes are parsed."""
        for block_type, codes in _VALID_CODES.items():
            for code in codes:
                block_type(code=code)

    def test_invalid(self):
        """Test if invalid codes raise exceptions."""
        for block_type, codes in _INVALID_CODES.items():
            for code in codes:
                with self.assertRaises(UnrecognizedCodeError):
                    block_type(code=code)

    def test_apply(self):
        """Test if the valid blocks can be applied."""

        block_ix = 0
        shape = [7, 31, 11]

        with tf.Session() as ses:
            for block_type, codes in _VALID_CODES.items():
                for code in codes:
                    block = block_type(code=code)
                    with tf.variable_scope('block_{}'.format(block_ix)):
                        x = tf.ones(shape)
                        y = block.apply(x)
                        self.assertEqual(len(y.shape), 3)
                        self.assertListEqual(y.get_shape().as_list()[:2], shape[:2])
                        self.assertNotEqual(y.get_shape().as_list()[2], shape[-1])
                    block_ix += 1

            self.assertEqual(37, RNNBlock(code='LSTM37').apply(x).get_shape().as_list()[-1])
            self.assertEqual(40, RNNBlock(code='biRNN20').apply(x).get_shape().as_list()[-1])
