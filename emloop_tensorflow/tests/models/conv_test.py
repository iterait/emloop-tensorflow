"""
Test module for the conv models.
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from unittest import TestCase
import emloop_tensorflow as eltf

from ..model_test import _OPTIMIZER


class SimpleAutoencoder(eltf.BaseModel):
    """
    Simple auto-encoder model.
    """

    def _create_model(self, use_bn=False) -> None:
        images = tf.placeholder(dtype=tf.float32, shape=(None, 100, 100, 3), name='images')
        masks = tf.placeholder(dtype=tf.float32, shape=(None, 100, 100), name='masks')
        _, net = eltf.models.cnn_autoencoder(images, ['8c3', 'mp2', '16inc', 'ap2', '32res'],
                                             is_training=self.is_training, use_bn=use_bn)
        net = slim.conv2d(net, 1, (5, 5), activation_fn=tf.nn.sigmoid, scope='cnn_final_inner')
        probabilities = tf.identity(net[:, :, :, 0], name='probabilities')
        pixel_loss = tf.losses.mean_squared_error(labels=masks, predictions=probabilities,
                                                  reduction=tf.losses.Reduction.NONE)
        tf.reduce_mean(pixel_loss, axis=(1, 2), name='loss')


class CNNEncoderTest(TestCase):
    """Test case for the cnn encoder."""

    def test_sanity(self):
        """Test cnn encoder sanitizes the arguments properly."""
        simple_encoder = ['12c3', '16inc', '12ress2', 'mp2']
        with tf.Graph().as_default(), tf.Session():
            is_training = tf.constant(False)
            x3 = tf.ones((10, 100, 100))
            x4 = tf.ones((10, 100, 100, 3))
            x6 = tf.ones((10, 2, 15, 100, 100, 3))
            with self.assertRaises(AssertionError):
                eltf.models.cnn_encoder(x3, simple_encoder, use_bn=False)  # 3-dim input is not supported
            with self.assertRaises(AssertionError):
                eltf.models.cnn_encoder(x6, simple_encoder, use_bn=False)  # 6-dim input is not supported
            with self.assertRaises(AssertionError):
                eltf.models.cnn_encoder(x4, simple_encoder, use_bn=True)  # missing is_training
            with self.assertRaises(AssertionError):  # both ln and bn at once
                eltf.models.cnn_encoder(x4, simple_encoder, is_training, use_bn=True, use_ln=True)

            with self.assertRaises(ValueError):
                eltf.models.cnn_encoder(x4, ['12c3', '12c3', 'dunno'], use_bn=False)

    def test_dims(self):
        """Test if the cnn encoder handles both 4 and 5 dim data."""
        with tf.Graph().as_default(), tf.Session() as ses:
            is_training = tf.constant(False)
            x4 = tf.ones((10, 100, 100, 3))
            x5 = tf.ones((10, 15, 100, 100, 3))
            for x in [x4, x5]:
                with tf.variable_scope('dim'+str(len(x.get_shape().as_list()))):
                    out = eltf.models.cnn_encoder(x, ['12c3', '16inc', '12ress2', 'mp2'], is_training)
                    with tf.variable_scope('same_dim'):
                        same_dim_out = eltf.models.cnn_encoder(x, ['12c3', '16inc', '12res', '3c3'], is_training)
                    ses.run(tf.local_variables_initializer())
                    ses.run(tf.global_variables_initializer())
                    value = out.eval(session=ses)
                    same_dim_value = same_dim_out.eval(session=ses)
                    self.assertIsInstance(value, np.ndarray)
                    self.assertEqual(value.ndim, len(x.get_shape().as_list()))
                    self.assertEqual(same_dim_value.shape, tuple(x.get_shape().as_list()))


class CNNAutoEncoderTest(TestCase):
    """Test case for the cnn auto-encoder."""

    def test_sanity(self):
        """Test cnn auto-encoder sanitizes the arguments properly."""
        simple_encoder = ['12c3', '16inc', '12ress2', 'mp2']
        with tf.Graph().as_default(), tf.Session():
            x3 = tf.ones((10, 100, 100))
            x4 = tf.ones((10, 100, 100, 3))
            x5 = tf.ones((10, 15, 100, 100, 3))
            with self.assertRaises(AssertionError):
                eltf.models.cnn_autoencoder(x3, simple_encoder)  # 3-dim input is not supported
            with self.assertRaises(AssertionError):
                eltf.models.cnn_autoencoder(x5, simple_encoder)  # 5-dim input is not supported
            with self.assertRaises(AssertionError):
                eltf.models.cnn_autoencoder(x4, ['mp2', '32c3'])  # first operation is pooling
            with tf.variable_scope('stride1'):
                with self.assertRaises(ValueError):
                    eltf.models.cnn_autoencoder(x4, ['32c3', '32c3s2'])  # strided block does not have inversion
            with tf.variable_scope('stride2'):
                with self.assertRaises(ValueError):
                    eltf.models.cnn_autoencoder(x4, ['32c3', '32ress2'])
            with tf.variable_scope('stride3'):
                with self.assertRaises(ValueError):
                    eltf.models.cnn_autoencoder(x4, ['32c3', 'ap3s2'])

    def test_padding(self):
        """Test cnn auto-encoder pads the input if needed and outputs the same shape anyways."""
        padding_encoder = ['3c3', 'mp3', '24c3', 'mp3', '48c3']
        fitted_encoder = ['3c3', 'mp2', '24c3', 'mp2', '48c3']  # pooling fits to the shape, no padding is required
        with tf.Graph().as_default(), tf.Session() as ses:
            x4 = tf.ones((10, 100, 100, 3))  # 100 is not divisible by 9 -> padding will be required
            with tf.variable_scope('padded'):
                _, decoded = eltf.models.cnn_autoencoder(x4, padding_encoder, use_bn=False)
            with tf.variable_scope('fitted'):
                encoded, _ = eltf.models.cnn_autoencoder(x4, fitted_encoder, use_bn=False)
            ses.run(tf.local_variables_initializer())
            ses.run(tf.global_variables_initializer())
            encoded_value = encoded.eval(session=ses)
            self.assertEqual(encoded_value.shape, (10, 25, 25, 48))
            decoded_value = decoded.eval(session=ses)
            self.assertEqual(decoded_value.shape, tuple(x4.get_shape().as_list()))

    def test_model_integration(self):
        """Test if cnn (auto)encoder is well integrated with eltf BaseModel."""
        model = SimpleAutoencoder(inputs=['images', 'masks'], outputs=['loss', 'probabilities'],
                                  dataset=None, optimizer=_OPTIMIZER, log_dir=None)
        outputs = model.run({'images': np.ones((3, 100, 100, 3)), 'masks': np.zeros((3, 100, 100), dtype=np.uint8)},
                            train=False)
        self.assertEqual(outputs['probabilities'].shape, (3, 100, 100))

        # test spanning across multiple GPUs
        multi_gpu_model = SimpleAutoencoder(inputs=['images', 'masks'], outputs=['loss', 'probabilities'],
                                            dataset=None, optimizer=_OPTIMIZER, log_dir=None, n_gpus=4,
                                            session_config={'allow_soft_placement': True})
        outputs2 = multi_gpu_model.run({'images': np.ones((3, 100, 100, 3)),
                                        'masks': np.zeros((3, 100, 100), dtype=np.uint8)}, train=True)
        self.assertEqual(outputs2['probabilities'].shape, (3, 100, 100))
