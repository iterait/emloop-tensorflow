"""
Test module for the loss ops.
"""

import numpy as np
import tensorflow as tf

from unittest import TestCase
from cxflow_tensorflow.ops import smooth_l1_loss


class SmoothL1Test(TestCase):
    """Test case for the smooth_l1_loss op."""

    def test_smooth_l1_loss(self):
        """ Test if ``smooth_l1_loss`` works properly."""

        with tf.Session().as_default():
            expected = tf.constant([[0, 0, 1]], dtype=tf.float32)
            predicted = tf.constant([[0.5, 1, 3]], dtype=tf.float32)
            expected_loss = np.array([[0.5*0.5*0.5, 0.5, 1.5]])
            computed_loss = smooth_l1_loss(expected, predicted).eval()

            self.assertTrue(np.all(expected_loss == computed_loss))
