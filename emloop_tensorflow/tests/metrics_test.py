"""
Test module for metrics util functions.
"""

import numpy as np
import tensorflow as tf

from emloop_tensorflow import bin_dice, bin_stats

_LABELS = [[1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0]]
_PREDICTIONS = [[0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]


def test_dice():
    """ Test if ``dice`` score is computed as expected."""
    with tf.Session().as_default():
        labels = tf.constant(_LABELS, dtype=tf.int32)
        predictions = tf.constant(_PREDICTIONS, dtype=tf.int32)
        expected_dice = [(2*1.) / (3+2), 0, 0]
        computed_dice = list(bin_dice(predictions, labels).eval())

        assert expected_dice == computed_dice[:-1]
        assert np.isnan(computed_dice[-1])


def test_stats():
    """ Test if f1, precision and recall stats are computed as expected."""
    with tf.Session().as_default():
        labels = tf.constant(_LABELS, dtype=tf.int32)
        predictions = tf.constant(_PREDICTIONS, dtype=tf.int32)
        expected_recall =    [1./3, 0., 0.,     np.nan]
        expected_precision = [1./2, 0., np.nan, np.nan]
        expected_f1 =        [0.4,  0., np.nan, np.nan]

        # test prefixless and suffixless
        f1_tensor, precision_tensor, recall_tensor = bin_stats(predictions, labels)
        assert f1_tensor.name == 'f1:0'
        assert precision_tensor.name == 'precision:0'
        assert recall_tensor.name == 'recall:0'

        np.testing.assert_equal(expected_f1, f1_tensor.eval())
        np.testing.assert_equal(expected_precision, precision_tensor.eval())
        np.testing.assert_equal(expected_recall, recall_tensor.eval())

        # test with prefix "silver" and suffix "gold"
        f1_named_tensor, precision_named_tensor, recall_named_tensor = bin_stats(predictions, labels,
                                                                                 prefix='silver', suffix='gold')
        assert f1_named_tensor.name == 'silver_f1_gold:0'
        assert precision_named_tensor.name == 'silver_precision_gold:0'
        assert recall_named_tensor.name == 'silver_recall_gold:0'

        np.testing.assert_equal(expected_f1, f1_named_tensor.eval())
        np.testing.assert_equal(expected_precision, precision_named_tensor.eval())
        np.testing.assert_equal(expected_recall, recall_named_tensor.eval())
