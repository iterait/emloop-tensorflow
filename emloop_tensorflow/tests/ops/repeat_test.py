"""
Test module for the repeat op.
"""

import numpy as np
import tensorflow as tf

from unittest import TestCase
from emloop_tensorflow.ops import repeat

class RepeastTest(TestCase):
    """Test case for the repeat op."""

    def test_repeat(self):
        """ Test if `repeat` works the same as np.repeat."""

        with tf.Session().as_default():
            # try different tensor types
            for npdtype, tfdtype in [(np.int32, tf.int32), (np.float32, tf.float32)]:
                for init_value in [np.array([0, 1, 2, 3], dtype=npdtype),
                                   np.array([[0, 1], [2, 3], [4, 5]], dtype=npdtype)]:
                    # and all their axes
                    for axis in range(len(init_value.shape)):
                        for repeats in [1, 2, 3, 11]:
                            tensor = tf.constant(init_value, dtype=tfdtype)

                            repeated_value = repeat(tensor, repeats=repeats, axis=axis).eval()
                            expected_value = np.repeat(init_value, repeats=repeats, axis=axis)

                            self.assertTrue(np.all(repeated_value == expected_value))
