"""
Test module for the get_coord_channels op.
"""

import numpy as np
import tensorflow as tf
import pytest

from emloop_tensorflow.ops import get_coord_channels


def test_get_coord_channels():
    """ Test if `get_coord_channels` works properly."""
    with tf.Session().as_default() as sess:
        inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='inputs')
        coords = get_coord_channels(inputs)

        inputs_shape = (7, 11, 13, 3)
        coords_np = sess.run(coords, feed_dict={inputs: np.zeros(inputs_shape)})
        assert coords_np.shape == inputs_shape[:3]+(2,)
        assert np.all(coords_np[:, :, 0, 0] == -1.)
        assert np.all(coords_np[:, -1, :, 1] == 1.)

        small_coords_np = sess.run(coords, feed_dict={inputs: np.zeros((1, 3, 3, 3))})
        small_coords_expected = np.array([[[[-1, -1], [0, -1], [1, -1]],
                                           [[-1,  0], [0,  0], [1,  0]],
                                           [[-1,  1], [0,  1], [1,  1]]]])
        assert np.array_equal(small_coords_np, small_coords_expected)


def test_get_coord_channels_dim():
    """ Test if `get_coord_channels` asserts dimensions as expected."""
    dim3 = tf.placeholder(tf.float32, shape=[None, None, 3], name='inputs3')
    dim5 = tf.placeholder(tf.float32, shape=[None, None, 3], name='inputs5')
    with pytest.raises(AssertionError):
        _ = get_coord_channels(dim3)
        _ = get_coord_channels(dim5)
