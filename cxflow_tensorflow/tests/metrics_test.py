"""
Test module for metrics util functions.
"""

import numpy as np
import tensorflow as tf

from cxflow.tests.test_core import CXTestCaseWithDir
from cxflow_tensorflow import bin_dice, bin_stats

_LABELS = [[1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0]]
_PREDICTIONS = [[0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]


class MetricsTest(CXTestCaseWithDir):
    """
    Test case for metrics functions.
    """

    def test_dice(self):
        """ Test if ``dice`` score is computed as expected."""

        with tf.Session().as_default():
            labels = tf.constant(_LABELS, dtype=tf.int32)
            predictions = tf.constant(_PREDICTIONS, dtype=tf.int32)
            expected_dice = [(2*1.) / (3+2), 0, 0]
            computed_dice = list(bin_dice(predictions, labels).eval())

            self.assertListEqual(expected_dice, computed_dice[:-1])
            self.assertTrue(np.isnan(computed_dice[-1]))

    def test_stats(self):
        """ Test if f1, precision and recall stats are computed as expected."""

        with tf.Session().as_default():
            labels = tf.constant(_LABELS, dtype=tf.int32)
            predictions = tf.constant(_PREDICTIONS, dtype=tf.int32)
            expected_recall =    [1./3, 0., 0.,     np.nan]
            expected_precision = [1./2, 0., np.nan, np.nan]
            expected_f1 =        [0.4,  0., np.nan, np.nan]
            computed_f1, computed_precision, computed_recall = \
                [computed.eval() for computed in bin_stats(predictions, labels)]

            np.testing.assert_equal(expected_f1, computed_f1)
            np.testing.assert_equal(expected_precision, computed_precision)
            np.testing.assert_equal(expected_recall, computed_recall)
