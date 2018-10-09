"""
Test module for sparse tensor ops.
"""

import numpy as np
import tensorflow as tf

from emloop_tensorflow.ops import dense_to_sparse


def test_dense_to_sparse():
    """ Test if `dense_to_sparse` works properly."""
    with tf.Session().as_default():
        dense = tf.constant([[1., 2., 0.], [0., 0., 3.]], dtype=tf.float32)

        sparse = dense_to_sparse(dense)

        assert np.array_equal(sparse.indices.eval(), np.array([[0, 0], [0, 1], [1, 2]]))
        assert np.array_equal(sparse.values.eval(), np.array([1., 2., 3.]))

        mask = tf.constant([[0, 1, 0], [1, 0, 0]], dtype=tf.int32)

        masked = dense_to_sparse(dense, mask)
        assert np.array_equal(masked.indices.eval(), np.array([[0, 1], [1, 0]]))
        assert np.array_equal(masked.values.eval(), np.array([2., 0.]))
