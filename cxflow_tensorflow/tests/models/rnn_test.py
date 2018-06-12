"""
Test module for the conv models.
"""
import numpy as np
import tensorflow as tf

from unittest import TestCase
import cxflow_tensorflow as cxtf

from ..model_test import _OPTIMIZER


class SimpleRNN(cxtf.BaseModel):
    """
    Simple RNN model.
    """

    def _create_model(self, use_sl: bool) -> None:
        sequences = tf.placeholder(dtype=tf.float32, shape=(None, 100, 32), name='sequences')
        sequence_lengths = tf.placeholder(dtype=tf.int32, shape=(None,), name='sequence_lengths')

        if use_sl:
            net = cxtf.models.rnn_stack(sequences, ['RNN64', 'RNN32', 'biRNN16'], sequence_length=sequence_lengths)
        else:
            net = cxtf.models.rnn_stack(sequences, ['RNN64', 'RNN32', 'biRNN16'])

        loss = tf.losses.mean_squared_error(labels=sequences, predictions=net, reduction=tf.losses.Reduction.NONE)
        tf.reduce_mean(loss, axis=(1, 2), name='loss')


class RNNSTackTest(TestCase):
    """Test case for the rnn stack."""

    def test_sanity(self):
        """Test rnn stack sanitizes the arguments properly."""
        stack = ['RNN64', 'RNN32', 'biRNN16']
        with tf.Graph().as_default(), tf.Session():
            x2 = tf.ones((7, 31))
            x3 = tf.ones((7, 31, 11))
            x4 = tf.ones((7, 31, 11, 13))
            seq_len = tf.ones((7, 31), tf.int32)
            with self.assertRaises(AssertionError):
                cxtf.models.rnn_stack(x2, stack)  # 2-dim input is not supported
            with self.assertRaises(AssertionError):
                cxtf.models.rnn_stack(x4, stack)  # 4-dim input is not supported
            with self.assertRaises(AssertionError):
                cxtf.models.rnn_stack(x3, stack, seq_len)  # sequence_length must be 1-dim

    def test_dims(self):
        """Test if the rnn stack handles data correctly."""
        with tf.Graph().as_default(), tf.Session() as ses:
            stack = ['RNN64', 'RNN32', 'biRNN16']
            x = tf.ones((7, 31, 11))
            sequence_length = tf.ones(7, tf.int32)
            for use_seq_len in [True, False]:
                with tf.variable_scope('sqln'+str(use_seq_len)):
                    if use_seq_len:
                        net = cxtf.models.rnn_stack(x, stack, sequence_length)
                    else:
                        net = cxtf.models.rnn_stack(x, stack)
                    ses.run(tf.local_variables_initializer())
                    ses.run(tf.global_variables_initializer())
                    value = net.eval(session=ses)
                    self.assertIsInstance(value, np.ndarray)
                    self.assertEqual(value.ndim, 3)

    def test_model_integration(self):
        """Test if cnn (auto)encoder is well integrated with cxtf BaseModel."""
        model = SimpleRNN(inputs=['sequences', 'sequence_lengths'], outputs=['loss'],
                          dataset=None, optimizer=_OPTIMIZER, log_dir=None, use_sl=True)

        outputs = model.run({'sequences': np.ones((3, 100, 32)), 'sequence_lengths': [42, 71, 13]}, train=False)
        self.assertEqual(outputs['loss'].shape, (3,))

        # test spanning across multiple GPUs
        multi_gpu_model = SimpleRNN(inputs=['sequences'], outputs=['loss'],
                                    dataset=None, optimizer=_OPTIMIZER, log_dir=None, n_gpus=4,
                                    session_config={'allow_soft_placement': True}, use_sl=False)
        outputs2 = multi_gpu_model.run({'sequences': np.ones((3, 100, 32))}, train=True)
        self.assertEqual(outputs2['loss'].shape, (3,))
