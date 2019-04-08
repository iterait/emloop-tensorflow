"""
Test module for the conv models implementation.
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pytest

from emloop_tensorflow.models.blocks import UnrecognizedCodeError, BaseBlock, Block
from emloop_tensorflow.models.conv_blocks import ConvBlock, ResBlock, IncBlock, AveragePoolBlock, \
    MaxPoolBlock, UnPoolBlock, GlobalAveragePoolBlock, CoordConvBlock

_VALID_CODES = {ResBlock: ['32res', '32ress2', '64ress3'],
                AveragePoolBlock: ['ap2', 'ap3s2'],
                GlobalAveragePoolBlock: ['gap'],
                MaxPoolBlock: ['mp2', 'mp3s2'],
                UnPoolBlock: ['up2', 'up3'],
                IncBlock: ['32inc', '64inc'],
                ConvBlock: ['32c3', '64c5s2', '64c3-5s2', '63c3-3'],
                CoordConvBlock: ['32cc3', '64cc3s2']}

_INVALID_CODES = {ResBlock: ['32ress', 'res', '32res2', '32residual', 'res32', '32res 32res 32res', 'res32s0', '12inc'],
                  AveragePoolBlock: ['map2', 'up3s2', '3ap', 'mp', 'mp2 32c3', 'mp0', '3c', '24ress2', 'mp2', 'up3'],
                  GlobalAveragePoolBlock: ['map2', 'up3s2', '3ap', 'mp', 'mp2 32c3', 'mp0',
                                           '3c', '24ress2', 'mp2', 'up3', 'ap', 'ga'],
                  MaxPoolBlock: ['map2', 'up3s2', '3ap', 'mp', 'mp2 32c3', 'mp0', '3c', '24ress2', 'ap2', 'up3'],
                  UnPoolBlock: ['map2', 'up3s2', '3ap', 'mp', 'mp2 32c3', 'mp0', '3c', '24ress2', 'mp2', 'ap3'],
                  IncBlock: ['32incs2', 'inc12', 'inception32', '64i', '54inc 54inc', '0inc', '32c'],
                  ConvBlock: ['32c', '32cs2', 'c3', '64cs', '32c3 32c3', '0c3', '3c0', 'mp2', '32c3-3-3'],
                  CoordConvBlock: ['cc3', '64cc', '0cc3']}


def test_sanity():
    """Test block concepts sanity."""
    with pytest.raises(NotImplementedError):
        Block('').apply(None)

    with pytest.raises(NotImplementedError):
        BaseBlock(code='a', regexp='.*')


def test_valid():
    """Test if valid codes are parsed."""
    for block_type, codes in _VALID_CODES.items():
        for code in codes:
            block_type(code=code)


def test_invalid():
    """Test if invalid codes raise exceptions."""
    for block_type, codes in _INVALID_CODES.items():
        for code in codes:
            with pytest.raises(UnrecognizedCodeError):
                block_type(code=code)


def test_apply():
    """Test if the valid blocks can be applied."""
    dim4_kwargs = {'extra_dim': (), 'ap_fn': slim.avg_pool2d, 'mp_fn': slim.max_pool2d, 'bn_fn': slim.batch_norm,
                   'ln_fn': tf.contrib.layers.layer_norm, 'conv_fn': slim.conv2d, 'pool_fn': slim.max_pool2d}
    dim5_kwargs = {'extra_dim': (1,), 'ap_fn': slim.avg_pool3d, 'mp_fn': slim.max_pool3d, 'bn_fn': slim.batch_norm,
                   'ln_fn': tf.contrib.layers.layer_norm, 'conv_fn': slim.conv3d, 'pool_fn': slim.max_pool3d}

    only5dim = ['64c3-5s2', '63c3-3']

    block_ix = 0
    for shape, kwargs in (((3, 100, 100, 3), dim4_kwargs), ((3, 13, 100, 100, 3), dim5_kwargs)):
        with tf.Session() as ses:
            for block_type, codes in _VALID_CODES.items():
                for code in codes:
                    if block_type == CoordConvBlock and len(shape) == 5:
                        continue  # coord conv does not work for 5dim tensors
                    block = block_type(code=code, **kwargs)
                    with tf.variable_scope('block_{}'.format(block_ix)):
                        x = tf.ones(shape)
                        if len(shape) == 5 or code not in only5dim:
                            x = block.apply(x)
                            ses.run(tf.local_variables_initializer())
                            ses.run(tf.global_variables_initializer())
                            value = x.eval(session=ses)
                            assert isinstance(value, np.ndarray)
                            if block_type == GlobalAveragePoolBlock:
                                assert value.ndim == len(shape)-2
                            else:
                                assert value.ndim == len(shape)
                            block_ix += 1
                        else:
                            with pytest.raises(ValueError):
                                block.apply(x)


def test_gap():
    """Test gap has no inverse code."""
    gap = GlobalAveragePoolBlock(code='gap')
    with pytest.raises(ValueError):
        _ = gap.inverse_code()
