"""Module with tf sparse tensor utilities."""
from typing import Optional

import tensorflow as tf


def dense_to_sparse(inputs: tf.Tensor, mask: Optional[tf.Tensor]=None) -> tf.SparseTensor:
    """
    Convert the given ``inputs`` tensor to a ``SparseTensor`` of its non-zero values.

    Optionally, use the given mask tensor for determining the values to be included in the ``SparseTensor``.

    :param inputs: input dense tensor
    :param mask: optional mask tensor
    :return: sparse tensor of non-zero (or masked) values
    """
    idx = tf.where(tf.not_equal((mask if mask is not None else inputs), 0))
    return tf.SparseTensor(idx, tf.gather_nd(inputs, idx), tf.shape(inputs, out_type=tf.int64))
