import tensorflow as tf


def get_last_valid_features(features: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor:
    """
    Get the last valid values from the given feature sequences.

    :param features: 3-dim batch-major tensor [batch, max_time, features]
    :param sequence_lengths: 1-dim tensor with sequence_lengths
    :return: last valid features, 2-dim tensor [batch, features]
    """
    assert len(features.shape) == 3, "Features must be 3-dim"
    assert len(sequence_lengths.shape) == 1, "Sequence lengths must be 1-dim"

    last_valid_indices = tf.stack((tf.range(tf.shape(sequence_lengths)[0]), sequence_lengths - 1), axis=-1)
    last_valid_features = tf.gather_nd(features, last_valid_indices)

    return last_valid_features
