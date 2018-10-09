import logging
from typing import Optional, Sequence

import tensorflow as tf

from .blocks import get_block_instance
from .rnn_blocks import RNNBlock

RNN_BLOCKS = [RNNBlock]
"""RNN blocks recognized by the functions in the ``rnn`` module."""

__all__ = ['rnn_stack', 'RNN_BLOCKS']


def rnn_stack(x: tf.Tensor,
              stack_config: Sequence[str],
              sequence_length: Optional[tf.Tensor]=None) -> tf.Tensor:
    """
    Build a recurrent neural network stack from the given ``stack_config`` (sequence of block codes).

    At the moment, the following blocks are recognized:

    +-------------+---------------------+------------------+
    |             | code                | example          |
    |-------------|---------------------|------------------|
    | Vanilla RNN | [bi]RNN(num_units)  | biRNN64, RNN128  |
    | GRU         | [bi]GRU(num_units)  | biGRU32, GRU64   |
    | LSTM        | [bi]LSTM(num_units) | biLSTM32, LSTM64 |
    +-------------+---------------------+------------------+

    References:

    - `RNNCells <https://www.tensorflow.org/api_guides/python/contrib.rnn>`_

    :param x: 3-dim batch-major input tensor [batch, max_time, features]
    :param stack_config: a sequence of RNN layer codes defining the stack architecture
    :param sequence_length: optional tensor with sequence lengths for better performance
    :return: 3-dim batch-major output of the rnn stack
    :raise UnrecognizedCodeError: if some of the layer configs cannot be correctly parsed
    :raise AssertionError: if the input tensor is not 3 dim
    """
    assert len(x.shape) == 3, 'RNN stack supports only 3 dim tensors'
    if sequence_length is not None:
        assert len(sequence_length.shape) == 1
    layer_kwargs = {'sequence_length': sequence_length}

    # batch-major to time-major (for performance reasons)
    x = tf.transpose(x, [1, 0, 2])

    for i, code in enumerate(stack_config):
        block, block_type = get_block_instance(code, RNN_BLOCKS, layer_kwargs)
        logging.debug('\tApplying `%s` block', block_type.__name__)
        with tf.variable_scope('{}_{}'.format(block_type.__name__, i)):
            x = block.apply(x)
        logging.debug('\t%s', x.get_shape())

    # back to batch-major
    x = tf.transpose(x, [1, 0, 2])
    return x
