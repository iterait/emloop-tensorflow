"""Module with flatten3D op."""
import numpy as np
import tensorflow as tf

def flatten3D(inputs: tf.Tensor) -> tf.Tensor:
    """
    Flatten the given ``inputs`` tensor to 3 dimensions.

    :param inputs: >=4d tensor to be flattened
    :return: 3d flatten tensor
    """
    assert len(inputs.get_shape().as_list()) > 3
    return tf.reshape(inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], np.prod(inputs.get_shape().as_list()[2:])])
