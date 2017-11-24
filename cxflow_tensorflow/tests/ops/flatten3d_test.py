"""
Test module for the flatten3D op.
"""

import numpy as np
import tensorflow as tf

from unittest import TestCase
from cxflow_tensorflow.ops import flatten3D

class Flatten3DTest(TestCase):
    """Test case for the flatten3D op."""

    def test_flatten3D(self):
        """ Test if `flatten3D` works properly."""

        with tf.Session().as_default():
            tensor2d = tf.constant([[1, 2, 3]], dtype=tf.float32)
            with self.assertRaises(AssertionError):
                output = flatten3D(tensor2d)

            init_shape = (3, 17, 23, 3, 5)
            expected_shape = (3, 17, 23*3*5)

            tensor5d = tf.constant(np.arange(0, np.prod(init_shape)).reshape(init_shape), tf.int32)
            self.assertEqual(tensor5d.eval().shape, init_shape)
            output = flatten3D(tensor5d)
            self.assertEqual(output.eval().shape, expected_shape)
