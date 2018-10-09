import numpy as np
import tensorflow as tf


def repeat(tensor: tf.Tensor, repeats: int, axis: int) -> tf.Tensor:
    """
    Repeat elements of the input tensor in the specified axis ``repeats``-times.

    .. note::
        Chaining of this op may produce TF warnings although the performance seems to be unaffected.

    :param tensor: TF tensor to be repeated
    :param repeats: number of repeats
    :param axis: axis to repeat
    :return: tensor with repeated elements
    """
    shape = tensor.get_shape().as_list()

    dims = np.arange(len(tensor.shape))
    prepare_perm = np.hstack(([axis], np.delete(dims, axis)))
    restore_perm = np.hstack((dims[1:axis+1], [0], dims[axis+1:]))

    indices = tf.cast(tf.floor(tf.range(0, shape[axis]*repeats)/tf.constant(repeats)), 'int32')

    shuffled = tf.transpose(tensor, prepare_perm)
    repeated = tf.gather(shuffled, indices)
    return tf.transpose(repeated, restore_perm)
