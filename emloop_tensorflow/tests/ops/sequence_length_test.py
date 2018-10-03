import tensorflow as tf
import numpy as np

from unittest import TestCase
from emloop_tensorflow.ops import get_last_valid_features


class LastValidFeatureTest(TestCase):
    """Test case for the get_last_valid_features op."""

    def test_valid_input(self):
        """Test if outputs are correct."""
        with tf.Session().as_default():
            features = tf.constant(np.arange(500).reshape((5, 100, 1)), dtype=tf.float32)
            sequence_lengths = tf.constant([1, 25, 50, 75, 100], dtype=tf.int32)
            expected_outputs = np.array([[0], [124], [249], [374], [499]], dtype=np.float32)
            outputs = get_last_valid_features(features, sequence_lengths)
            outputs = outputs.eval()
            self.assertTrue(np.allclose(expected_outputs, outputs))

    def test_sequence_length_invalid_value(self):
        """Test invalid sequence_lengths values raise ValueError."""
        with tf.Session().as_default():
            features = tf.constant(np.arange(100).reshape((1, 100, 1)), dtype=tf.float32)
            sequence_lengths_0 = tf.constant([0], dtype=tf.int32)
            sequence_lengths_101 = tf.constant([101], dtype=tf.int32)
            outputs_0 = get_last_valid_features(features, sequence_lengths_0)
            outputs_101 = get_last_valid_features(features, sequence_lengths_101)
            with self.assertRaises(tf.errors.InvalidArgumentError):
                outputs_0.eval()
            with self.assertRaises(tf.errors.InvalidArgumentError):
                outputs_101.eval()

    def test_invalid_dims(self):
        """Test invalid dims raise AssertionError."""
        features_3d = tf.constant(np.arange(200).reshape((2, 100, 1)), dtype=tf.float32)
        sequence_lengths_2d = tf.constant(np.array([[1], [2]], dtype=np.int32))
        features_4d = tf.constant(np.arange(200).reshape((2, 100, 1, 1)), dtype=tf.float32)
        sequence_lengths_1d = tf.constant([1, 2], dtype=np.int32)
        with self.assertRaises(AssertionError):
            get_last_valid_features(features_3d, sequence_lengths_2d)
        with self.assertRaises(AssertionError):
            get_last_valid_features(features_4d, sequence_lengths_1d)
