"""
Module with tf util functions computing various ml metrics.
"""
from typing import Tuple, Optional

import tensorflow as tf


def bin_stats(predictions: tf.Tensor, labels: tf.Tensor, prefix: Optional[str]=None,
              suffix: Optional[str]=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Calculate f1, precision and recall from binary classification expected and predicted values.

    :param predictions: 2-d tensor (batch, predictions) of predicted 0/1 classes
    :param labels: 2-d tensor (batch, labels) of expected 0/1 classes
    :param prefix: prefix of the output tensor names
    :param suffix: suffix of the output tensor names
    :return: a tuple of batched (f1, precision and recall) values
    """

    # build correct prefix and suffix
    if suffix is None:
        suffix = ''
    else:
        suffix = '_' + suffix

    if prefix is None:
        prefix = ''
    else:
        prefix = prefix + '_'

    # compute bin. statistics
    predictions = tf.cast(predictions, tf.int32)
    labels = tf.cast(labels, tf.int32)

    true_positives = tf.reduce_sum((predictions * labels), axis=1)
    false_positives = tf.reduce_sum(tf.cast(tf.greater(predictions, labels), tf.int32), axis=1)
    false_negatives = tf.reduce_sum(tf.cast(tf.greater(labels, predictions), tf.int32), axis=1)

    recall = tf.identity(true_positives / (true_positives + false_negatives),
                         name=prefix+'recall'+suffix)
    precision = tf.identity(true_positives / (true_positives + false_positives),
                            name=prefix+'precision'+suffix)
    f1_score = tf.identity(2 / (1 / precision + 1 / recall),
                           name=prefix+'f1'+suffix)

    return f1_score, precision, recall


def bin_dice(predictions: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """
    Calculate Sorensen–Dice coefficient from the given binary classification expected and predicted values.

    The coefficient is defined as :math:`2*|X \cup Y| / (|X| + |Y|)`.

    :param predictions: 2-d tensor (batch, predictions) of predicted 0/1 classes
    :param labels: 2-d tensor (batch, labels) of expected 0/1 classes
    :return: batched Sørensen–Dice coefficients
    """
    predictions = tf.cast(predictions, tf.int32)
    labels = tf.cast(labels, tf.int32)

    true_positives = tf.reduce_sum((predictions * labels), axis=1)
    pred_positives = tf.reduce_sum(predictions, axis=1)
    label_positives = tf.reduce_sum(labels, axis=1)

    return 2 * true_positives / (pred_positives + label_positives)
