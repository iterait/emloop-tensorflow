"""
Test module for the flatten3D op.
"""

import numpy as np
import tensorflow as tf
import pytest

from emloop_tensorflow.ops import flatten3D


def test_flatten3D():
    """ Test if `flatten3D` works properly."""
    with tf.Session().as_default():
        tensor2d = tf.constant([[1, 2, 3]], dtype=tf.float32)
        with pytest.raises(AssertionError):
            output = flatten3D(tensor2d)

        tensor3d = tf.constant([[[1, 2, 3]]], dtype=tf.float32)
        assert tensor3d == flatten3D(tensor3d)

        init_shape = (3, 17, 23, 3, 5)
        expected_shape = (3, 17, 23*3*5)

        tensor5d = tf.constant(np.arange(0, np.prod(init_shape)).reshape(init_shape), tf.int32)
        assert tensor5d.eval().shape == init_shape
        output = flatten3D(tensor5d)
        assert output.eval().shape == expected_shape
