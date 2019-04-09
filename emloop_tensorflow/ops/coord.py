"""Module with get_coord_channels op."""
import tensorflow as tf


def get_coord_channels(inputs: tf.Tensor) -> tf.Tensor:
    """
    For the given 4-dim tensor (supposedly batch x height x width x channels) return two extra channels with
    x and y coordinates scaled to [-1, 1] interval.
    I.e.:
        - ``channels[:, :, i, 0] == [SCALED i]`` while
        - ``channels[:, j, :, 1] == [SCALED j]``.

    :param inputs: 4d tensor
    :return: two extra channels with scaled coordinates
    """
    assert len(inputs.get_shape().as_list()) == 4
    shape = tf.shape(inputs)
    x = tf.linspace(-1., 1., shape[2])
    y = tf.linspace(-1., 1., shape[1])

    x = tf.broadcast_to(x, shape[:3])
    y = tf.broadcast_to(y, [shape[0], shape[2], shape[1]])
    y = tf.transpose(y, [0, 2, 1])

    coords = tf.stack([x, y], axis=-1)

    return coords
