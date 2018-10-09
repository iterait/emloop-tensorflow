from typing import Optional

import tensorflow as tf

from .blocks import BaseBlock

__all__ = ['RNNBlock']


class RNNBlock(BaseBlock):
    """Base block for convolution-like blocks."""

    def __init__(self, sequence_length: Optional[tf.Tensor]=None, **kwargs):
        """
        Try to parse and create new :py:class:`RNNBlock`.
        """
        self._num_units = self._cell_type = self._bidirectional = None
        self._sequence_length = sequence_length
        super().__init__(regexp='(bi)?(LSTM|GRU|RNN)([1-9][0-9]*)',
                         defaults=(False, None, None),
                         **kwargs)

    def _handle_parsed_args(self, bidirectional: Optional[str], cell_type: str, num_units: int) -> None:
        self._bidirectional = bidirectional == 'bi'
        self._num_units = int(num_units)
        self._cell_type = cell_type
        if self._cell_type in ['RNN', 'LSTM']:
            self._cell_type = 'Basic'+self._cell_type
        self._cell_type += 'Cell'

    def apply(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply the RNN block to the given input.

        :param x: 3-dim time-major sequences of features [max_time, batch, features]
        :return: output 3-dim tensor [max_time, batch, transformed_features]
        """
        cell_fn = getattr(tf.contrib.rnn, self._cell_type)

        if self._bidirectional:
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fn(num_units=self._num_units),
                                                         cell_bw=cell_fn(num_units=self._num_units),
                                                         inputs=x,
                                                         sequence_length=self._sequence_length,
                                                         time_major=True,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell=cell_fn(num_units=self._num_units),
                                           inputs=x,
                                           sequence_length=self._sequence_length,
                                           time_major=True,
                                           dtype=tf.float32)

        return outputs
