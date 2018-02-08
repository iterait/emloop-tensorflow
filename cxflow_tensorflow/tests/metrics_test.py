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

            # test suffixless
            f1_tensor, precision_tensor, recall_tensor = bin_stats(predictions, labels)
            self.assertEqual(f1_tensor.name, 'f1:0')
            self.assertEqual(precision_tensor.name, 'precision:0')
            self.assertEqual(recall_tensor.name, 'recall:0')

            np.testing.assert_equal(expected_f1, f1_tensor.eval())
            np.testing.assert_equal(expected_precision, precision_tensor.eval())
            np.testing.assert_equal(expected_recall, recall_tensor.eval())

            # test with suffix "gold"
            f1_named_tensor, precision_named_tensor, recall_named_tensor = bin_stats(predictions, labels, suffix='gold')
            self.assertEqual(f1_named_tensor.name, 'f1_gold:0')
            self.assertEqual(precision_named_tensor.name, 'precision_gold:0')
            self.assertEqual(recall_named_tensor.name, 'recall_gold:0')

            np.testing.assert_equal(expected_f1, f1_named_tensor.eval())
            np.testing.assert_equal(expected_precision, precision_named_tensor.eval())
            np.testing.assert_equal(expected_recall, recall_named_tensor.eval())
