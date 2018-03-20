import logging
from typing import Optional, Callable, Sequence, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .blocks import get_block_instance
from .conv_blocks import ConvBlock, IncBlock, ResBlock, MaxPoolBlock, AveragePoolBlock, UnPoolBlock

CONV_BLOCKS = [ConvBlock, IncBlock, ResBlock, MaxPoolBlock, AveragePoolBlock, UnPoolBlock]
"""CNN blocks recognized by the functions in the ``conv`` module."""

POOL_BLOCKS = [MaxPoolBlock, AveragePoolBlock]
"""Pooling blocks to be reversed in the ``cnn_autoencoder`` function."""

UNPOOL_BLOCK = UnPoolBlock
"""Unpooling block (inverse to the pooling blocks)."""


__all__ = ['cnn_encoder', 'cnn_autoencoder', 'compute_pool_amount', 'CONV_BLOCKS', 'POOL_BLOCKS', 'UNPOOL_BLOCK']


def compute_pool_amount(encoder_config: Sequence[str]):
    """
    Compute the amount of pooling in the given ``encoder_config``.
    E.g.: with two max pool layers with kernel size 2, the outputs would be 4.

    :param encoder_config: a sequence of CNN encoder config codes
    :return: the amount of pooling in the given ``encoder_config``
    """
    blocks_and_types = [get_block_instance(code, CONV_BLOCKS) for code in encoder_config]
    return np.prod([block.kernel_size for block, block_type in blocks_and_types if block_type in POOL_BLOCKS])


def cnn_encoder(x: tf.Tensor,
                encoder_config: Sequence[str],
                is_training: Optional[tf.Tensor]=None,
                activation: Callable[[tf.Tensor], tf.Tensor]=tf.nn.relu,
                conv_kwargs: Optional[dict]=None,
                bn_kwargs: Optional[dict]=None,
                ln_kwargs: Optional[dict]=None,
                skip_connections: Optional[List[tf.Tensor]]=None,
                use_bn: bool=False,
                use_ln: bool=False) -> tf.Tensor:
    """
    Build a convolutional neural network from the given ``encoder_config`` (sequence of block codes).

    At the moment, the following blocks are recognized:

    +---------------------------+-------------------------------------------------------------+------------------------+
    |                           | code                                                        | example                |
    +---------------------------+-------------------------------------------------------------+------------------------+
    | Convolutional layer       | (num_filters)c[(time_kernel_size)-](kernel_size)[s(stride)] | 64c3, 64c3s2, 64c3-3s2 |
    +---------------------------+-------------------------------------------------------------+------------------------+
    | Average/Max pooling layer | (ap|mp)(kernel_size)[s(stride)]                             | mp2, ap3s2             |
    +---------------------------+-------------------------------------------------------------+------------------------+
    | Inception block           | (num_filters)inc                                            | 128inc                 |
    +---------------------------+-------------------------------------------------------------+------------------------+
    | Residual block            | (num_filters)res[s(stride)]                                 | 512ress2               |
    +---------------------------+-------------------------------------------------------------+------------------------+

    .. code-block:: python
        :caption: Usage

        images = tf.placeholder(dtype=tf.float32, shape=(None, height, width, channels), name='images')
        net = cnn_encoder(images, ['64c7s2', '128inc', '128inc', 'ap3s2', '256inc', '256inc'],
                          self.is_training, tf.nn.elu)
        # use encoded features here

    .. tip::
        CNN encoder can be applied to both 4D and 5D tensors.

    **Skip connections:**

    When provided, this function appends the *pre-pooling* tensors to the ``skip_connections`` sequence. Inversely, skip
    connections are popped and added to the *post-unpooling* tensors if they are available.

    References:

    - `ResNet <https://arxiv.org/pdf/1512.03385.pdf>`_
    - `InceptionNet <https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf>`_
    - `Batch normalization <http://proceedings.mlr.press/v37/ioffe15.pdf>`_
    - `Layer normalization <https://arxiv.org/pdf/1607.06450.pdf>`_

    :param x: 4 or 5 dim input tensor
    :param encoder_config: sequence of layer/block codes defining the CNN architecture
    :param is_training: training/eval phase indicator
    :param activation: activation function
    :param conv_kwargs: convolutional layers kwargs
    :param bn_kwargs: batch normalization layers kwargs
    :param ln_kwargs: layer normalization layers kwargs
    :param skip_connections: mutable sequence of skip connection tensors around pooling operations
    :param use_bn: add batch normalization layers after each convolution (including res and inception modules)
    :param use_ln: add layer normalization layers after each convolution/module
    :return: output tensor of the specified CNN when applied to the given input tensor
    :raise UnrecognizedCodeError: if some of the layer configs cannot be correctly parsed
    :raise AssertionError: if the input tensor is not 4 nor 5 dim
    """
    assert len(x.get_shape()) in [4, 5], 'CNN encoder supports only 4 or 5 dim tensors'
    assert not use_bn or not use_ln
    if use_bn:
        assert is_training is not None, '`is_training` flag must be provided when `use_bn` is true'

    # apply default arguments
    conv_kwargs = conv_kwargs or {}
    bn_kwargs = bn_kwargs or {}
    ln_kwargs = ln_kwargs or {}

    merged_conv_kwargs = {
        'activation_fn': activation,
        'padding': 'SAME'
    }
    merged_bn_kwargs = {
        'decay': 0.95,
        'is_training': is_training
    }
    for kwargs, user_kwargs in ((merged_conv_kwargs, conv_kwargs), (merged_bn_kwargs, bn_kwargs)):
        kwargs.update(user_kwargs)

    # include dev name in the BN and LN variable names in order to avoid inter-device dependencies
    dev = str(x.device).replace(':', ' ').replace('/', ' ').strip().replace(' ', '-').lower()

    def bn_fn(x_in: tf.Tensor, *args, **kwargs_):
        kwargs_.update(scope='bn-'+dev)
        return slim.batch_norm(x_in, *args, **kwargs_) if use_bn else x_in

    def ln_fn(x_in: tf.Tensor, *args, **kwargs_):
        final_kwargs = ln_kwargs.copy()
        final_kwargs.update(kwargs_)
        final_kwargs.update(scope='ln-'+dev)
        return tf.contrib.layers.layer_norm(x_in, *args, **final_kwargs) if use_ln else x_in

    extra_dim = (1,) if len(x.shape) == 5 else ()
    conv_fn = slim.conv3d if extra_dim else slim.conv2d
    mp_fn = slim.max_pool3d if extra_dim else slim.max_pool2d
    ap_fn = slim.avg_pool3d if extra_dim else slim.avg_pool2d

    block_kwargs = {'extra_dim': extra_dim, 'ap_fn': ap_fn, 'mp_fn': mp_fn, 'bn_fn': bn_fn, 'ln_fn': ln_fn,
                    'conv_fn': conv_fn, 'pool_fn': mp_fn}

    with slim.arg_scope([conv_fn], **merged_conv_kwargs), slim.arg_scope([slim.batch_norm], **merged_bn_kwargs):
        for i, code in enumerate(encoder_config):
            block, block_type = get_block_instance(code, CONV_BLOCKS, block_kwargs)
            logging.debug('\tApplying `%s` block', block_type.__name__)
            if skip_connections is not None and (block_type in POOL_BLOCKS):
                skip_connections.append(x)
            with tf.variable_scope('{}_{}'.format(block_type.__name__, i)):
                x = block.apply(x)
            if skip_connections is not None and (block_type is UNPOOL_BLOCK):
                x += skip_connections.pop()
            logging.debug('\t%s', x.get_shape())
    return x


def cnn_autoencoder(x: tf.Tensor,
                    encoder_config: Sequence[str],
                    is_training: Optional[tf.Tensor]=None,
                    activation: Callable[[tf.Tensor], tf.Tensor]=tf.nn.relu,
                    conv_kwargs: Optional[dict]=None,
                    bn_kwargs: Optional[dict]=None,
                    ln_kwargs: Optional[dict]=None,
                    skip_connections: bool=True,
                    use_bn: bool=False,
                    use_ln: bool=False) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Build a convolutional auto-encoder from the given ``encoder_config`` (sequence of layer/block codes).

    For the list of supported layers, modules etc. see :py:func:`cnn_encoder` function.

    The process of auto-encoder construction is as follows.

    1. Create a CNN encoder based on the encoder config.
    2. Create a CNN decoder based on the reversed encoder config (reversed order, un-pooling instead of pooling)

    .. warning::
        Does not support strided layers/modules and 5 dim tensors.

    :param x: 4 dim input tensor
    :param encoder_config: sequence of layer/block codes defining the CNN architecture
    :param is_training: training/eval phase indicator
    :param activation: activation function
    :param conv_kwargs: convolutional layers kwargs
    :param bn_kwargs: batch normalization layers kwargs
    :param ln_kwargs: layer normalization layers kwargs
    :param skip_connections: include encoder-decoder skip connections around pooling operations
    :param use_bn: add batch normalization layers after each convolution (including res and inception modules)
    :param use_ln: add layer normalization layers after each convolution/module
    :return: a tuple of encoded tensor and output (decoded) tensor
    :raise ValueError: if some of the layer configs cannot be correctly parsed
    :raise AssertionError: if the configuration does not meet the requirements
    """
    assert len(x.get_shape()) == 4, 'CNN auto-encoder supports only 4 dim tensors'
    assert get_block_instance(encoder_config[0], CONV_BLOCKS)[1] not in POOL_BLOCKS, \
        'You choose pooling as the first action in the auto-encoder. Resize your image ' \
        'and delete first pooling layer - you will save memory and gain the same result.'

    # save the original shape for further checks
    original_shape = x.get_shape().as_list()
    logging.debug('Constructing an auto-encoder for tensor of shape: %s', original_shape)

    # compute
    blocks_and_types = [get_block_instance(code, CONV_BLOCKS) for code in encoder_config]
    pool_product = compute_pool_amount(encoder_config)
    padded = False
    if pool_product > 1:
        rows, cols = x.get_shape().as_list()[1:3]
        target_rows = rows + pool_product - rows % pool_product if rows % pool_product > 0 else rows
        target_cols = cols + pool_product - cols % pool_product if cols % pool_product > 0 else cols
        if rows != target_rows or cols != target_cols:
            padded = True
            logging.debug('Padding from: %s', x.get_shape())

            pad_top = (target_rows - rows) // 2
            pad_bot = target_rows - pad_top - rows
            pad_left = (target_cols - cols) // 2
            pad_right = target_cols - pad_left - cols
            logging.debug('Paddings: left=%d, right=%d, top=%d, bottom=%d', pad_top, pad_bot, pad_left, pad_right)

            x = tf.concat([tf.zeros_like(x[:, :pad_top, :, :], dtype=x.dtype), x,
                           tf.zeros_like(x[:, :pad_bot, :, :], dtype=x.dtype)], axis=1)
            x = tf.concat([tf.zeros_like(x[:, :, :pad_left, :], dtype=x.dtype), x,
                           tf.zeros_like(x[:, :, :pad_right, :], dtype=x.dtype)], axis=2)

            logging.debug('\t%s', x.get_shape())

    skip_connections = [] if skip_connections else None
    # build an encoder
    with tf.variable_scope('encoder'):
        encoded = cnn_encoder(x, encoder_config, is_training, activation, conv_kwargs, bn_kwargs, ln_kwargs,
                              skip_connections, use_bn, use_ln)
    # re-arrange the pooling layers so that the decoder is symmetrical
    if len(encoder_config) > 1:
        for i, (_, block_type) in enumerate(blocks_and_types):
            if block_type in POOL_BLOCKS:
                encoder_config[i - 1], encoder_config[i] = encoder_config[i], encoder_config[i - 1]

    decoder_config = [get_block_instance(code, CONV_BLOCKS)[0].inverse_code() for code in reversed(encoder_config)]
    with tf.variable_scope('decoder'):
        decoded_raw = cnn_encoder(encoded, decoder_config, is_training, activation, conv_kwargs, bn_kwargs, ln_kwargs,
                                  skip_connections, use_bn, use_ln)

    logging.debug('Shape of the padded auto-encoder: %s', decoded_raw.get_shape())
    decoded = decoded_raw[:, pad_top:-pad_bot, pad_left:-pad_right, :] if padded else decoded_raw
    logging.debug('Shape of the final slice: %s', decoded.get_shape())

    return encoded, decoded
