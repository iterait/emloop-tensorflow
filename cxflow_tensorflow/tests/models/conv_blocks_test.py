"""
Test module for the conv models implementation.
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from unittest import TestCase
from cxflow_tensorflow.models.blocks import UnrecognizedCodeError, BaseBlock, Block
from cxflow_tensorflow.models.conv_blocks import ConvBlock, ResBlock, IncBlock, AveragePoolBlock, \
    MaxPoolBlock, UnPoolBlock

_VALID_CODES = {ResBlock: ['32res', '32ress2', '64ress3'],
                AveragePoolBlock: ['ap2', 'ap3s2'],
                MaxPoolBlock: ['mp2', 'mp3s2'],
                UnPoolBlock: ['up2', 'up3'],
                IncBlock: ['32inc', '64inc'],
                ConvBlock: ['32c3', '64c5s2']}

_INVALID_CODES = {ResBlock: ['32ress', 'res', '32res2', '32residual', 'res32', '32res 32res 32res', 'res32s0', '12inc'],
                  AveragePoolBlock: ['map2', 'up3s2', '3ap', 'mp', 'mp2 32c3', 'mp0', '3c', '24ress2', 'mp2', 'up3'],
                  MaxPoolBlock: ['map2', 'up3s2', '3ap', 'mp', 'mp2 32c3', 'mp0', '3c', '24ress2', 'ap2', 'up3'],
                  UnPoolBlock: ['map2', 'up3s2', '3ap', 'mp', 'mp2 32c3', 'mp0', '3c', '24ress2', 'mp2', 'ap3'],
                  IncBlock: ['32incs2', 'inc12', 'inception32', '64i', '54inc 54inc', '0inc', '32c'],
                  ConvBlock: ['32c', '32cs2', 'c3', '64cs', '32c3 32c3', '0c3', '3c0', 'mp2']}


class BlocksTest(TestCase):
    """Test conv blocks parsing and applying."""

    def test_sanity(self):
        """Test block concepts sanity."""
        with self.assertRaises(NotImplementedError):
            Block('').apply(None)

        with self.assertRaises(NotImplementedError):
            BaseBlock(code='a', regexp='.*')

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
        dim4_kwargs = {'extra_dim': (), 'ap_fn': slim.avg_pool2d, 'mp_fn': slim.max_pool2d, 'bn_fn': slim.batch_norm,
                       'ln_fn': tf.contrib.layers.layer_norm, 'conv_fn': slim.conv2d, 'pool_fn': slim.max_pool2d}
        dim5_kwargs = {'extra_dim': (1,), 'ap_fn': slim.avg_pool3d, 'mp_fn': slim.max_pool3d, 'bn_fn': slim.batch_norm,
                       'ln_fn': tf.contrib.layers.layer_norm, 'conv_fn': slim.conv3d, 'pool_fn': slim.max_pool3d}

        block_ix = 0
        for shape, kwargs in (((3, 100, 100, 3), dim4_kwargs), ((3, 13, 100, 100, 3), dim5_kwargs)):
            with tf.Session() as ses:
                for block_type, codes in _VALID_CODES.items():
                    for code in codes:
                        block = block_type(code=code, **kwargs)
                        with tf.variable_scope('block_{}'.format(block_ix)):
                            x = tf.ones(shape)
                            x = block.apply(x)
                            ses.run(tf.local_variables_initializer())
                            ses.run(tf.global_variables_initializer())
                            value = x.eval(session=ses)
                            self.assertIsInstance(value, np.ndarray)
                            self.assertEqual(value.ndim, len(shape))
                            block_ix += 1
