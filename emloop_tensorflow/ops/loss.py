"""Module with various custom loss ops."""
import tensorflow as tf


def smooth_l1_loss(predicted: tf.Tensor, expected: tf.Tensor) -> tf.Tensor:
    """
    Calculate piece-wise smooth L1 loss on the given tensors.

    Reference: `Fast R-CNN <https://arxiv.org/pdf/1504.08083.pdf>`_

    :param predicted: predicted values tensor
    :param expected: expected values tensor with the same shape as the ``predicted`` tensor
    :return: piece-wise smooth L1 loss
    """
    abs_diff = tf.abs(predicted - expected)
    return tf.where(tf.less(abs_diff, 1), 0.5 * tf.square(abs_diff), abs_diff - 0.5)
