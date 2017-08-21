"""
Test module for metrics util functions.
"""

import numpy as np
import tensorflow as tf

from cxflow.tests.test_core import CXTestCaseWithDir
from cxflow_tensorflow import bin_dice, bin_stats

_LABELS = [[1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0]]
_PREDICTIONS = [[0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0]]


class MetricsTest(CXTestCaseWithDir):
    """
    Test case for metrics functions.
    """

    def test_dice(self):
        """ Test if `dice` score is computed as expected."""

        with tf.Session().as_default():
            labels = tf.constant(_LABELS, dtype=tf.int32)
            predictions = tf.constant(_PREDICTIONS, dtype=tf.int32)
            expected_dice = np.array([(2*1.) / (3+2), 0])
            computed_dice = bin_dice(predictions, labels).eval()

            self.assertTrue(np.all(expected_dice == computed_dice[:-1]))
            self.assertTrue(np.isnan(computed_dice[-1]))

    def test_stats(self):
        """ Test if f1, precision and recall stats are computed as expected."""

        with tf.Session().as_default():
            labels = tf.constant(_LABELS, dtype=tf.int32)
            predictions = tf.constant(_PREDICTIONS, dtype=tf.int32)
            expected_recall = np.array([1./3, 0.])
            expected_precision = np.array([1./2, 0.])
            expected_f1 = 2/(1/expected_precision + 1/expected_recall)
            computed_f1, computed_precision, computed_recall = \
                [computed.eval() for computed in bin_stats(predictions, labels)]

            self.assertTrue(np.all(expected_f1 == computed_f1[:-1]))
            self.assertTrue(np.all(expected_precision == computed_precision[:-1]))
            self.assertTrue(np.all(expected_recall == computed_recall[:-1]))
            for computed in (computed_f1, computed_precision, computed_recall):
                self.assertTrue(np.isnan(computed[-1]))
