"""
Test module for the conv models.
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pytest

import emloop_tensorflow as eltf

from ..model_test import _OPTIMIZER


class SimpleAutoencoder(eltf.BaseModel):
    """Simple auto-encoder model."""

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


###############
# CNN Encoder #
###############
"""Test case for the cnn encoder."""


def test_sanity():
    """Test cnn encoder sanitizes the arguments properly."""
    simple_encoder = ['12c3', '16inc', '12ress2', 'mp2']
    with tf.Graph().as_default(), tf.Session():
        is_training = tf.constant(False)
        x3 = tf.ones((10, 100, 100))
        x4 = tf.ones((10, 100, 100, 3))
        x6 = tf.ones((10, 2, 15, 100, 100, 3))
        with pytest.raises(AssertionError):
            eltf.models.cnn_encoder(x3, simple_encoder, use_bn=False)  # 3-dim input is not supported
        with pytest.raises(AssertionError):
            eltf.models.cnn_encoder(x6, simple_encoder, use_bn=False)  # 6-dim input is not supported
        with pytest.raises(AssertionError):
            eltf.models.cnn_encoder(x4, simple_encoder, use_bn=True)  # missing is_training
        with pytest.raises(AssertionError):  # both ln and bn at once
            eltf.models.cnn_encoder(x4, simple_encoder, is_training, use_bn=True, use_ln=True)

        with pytest.raises(ValueError):
            eltf.models.cnn_encoder(x4, ['12c3', '12c3', 'dunno'], use_bn=False)


def test_dims():
    """Test if the cnn encoder handles both 4 and 5 dim data."""
    with tf.Graph().as_default(), tf.Session() as ses:
        is_training = tf.constant(False)
        x4 = tf.ones((10, 100, 100, 3))
        x5 = tf.ones((10, 15, 100, 100, 3))
        config_base = ['12c3', '16inc', '12ress2', 'mp2']
        for x, config in [(x4, ['3cc3']+config_base), (x5, config_base)]:
            with tf.variable_scope('dim'+str(len(x.get_shape().as_list()))):
                out = eltf.models.cnn_encoder(x, config, is_training)
                with tf.variable_scope('same_dim'):
                    config = config[:-2]+['3res']
                    same_dim_out = eltf.models.cnn_encoder(x, config, is_training)
                ses.run(tf.local_variables_initializer())
                ses.run(tf.global_variables_initializer())
                value = out.eval(session=ses)
                same_dim_value = same_dim_out.eval(session=ses)
                assert isinstance(value, np.ndarray)
                assert value.ndim == len(x.get_shape().as_list())
                assert same_dim_value.shape == tuple(x.get_shape().as_list())


###################
# CNN AutoEncoder #
###################
"""Test case for the cnn auto-encoder."""


def test_sanity_auto():
    """Test cnn auto-encoder sanitizes the arguments properly."""
    simple_encoder = ['12c3', '16inc', '12ress2', 'mp2']
    with tf.Graph().as_default(), tf.Session():
        x3 = tf.ones((10, 100, 100))
        x4 = tf.ones((10, 100, 100, 3))
        x5 = tf.ones((10, 15, 100, 100, 3))
        with pytest.raises(AssertionError):
            eltf.models.cnn_autoencoder(x3, simple_encoder)  # 3-dim input is not supported
        with pytest.raises(AssertionError):
            eltf.models.cnn_autoencoder(x5, simple_encoder)  # 5-dim input is not supported
        with pytest.raises(AssertionError):
            eltf.models.cnn_autoencoder(x4, ['mp2', '32c3'])  # first operation is pooling
        with tf.variable_scope('stride1'):
            with pytest.raises(ValueError):
                eltf.models.cnn_autoencoder(x4, ['32c3', '32c3s2'])  # strided block does not have inversion
        with tf.variable_scope('stride2'):
            with pytest.raises(ValueError):
                eltf.models.cnn_autoencoder(x4, ['32c3', '32ress2'])
        with tf.variable_scope('stride3'):
            with pytest.raises(ValueError):
                eltf.models.cnn_autoencoder(x4, ['32c3', 'ap3s2'])


# The image sizes for the following are not divisible by 9, but they are divisible by 4.
@pytest.mark.parametrize("input_shape,padding_encoded_shape,fitted_encoded_shape", [
    ((10, 100, 100, 3), (10, 12, 12, 48), (10, 25, 25, 48)),
    ((10, 40, 80, 3), (10, 5, 9, 48), (10, 10, 20, 48)),
])
def test_padding(input_shape, padding_encoded_shape, fitted_encoded_shape):
    """Test cnn auto-encoder pads the input if needed and outputs the same shape anyways."""
    padding_encoder = ['3c3', 'mp3', '24c3', 'mp3', '48c3']
    fitted_encoder = ['3c3', 'mp2', '24c3', 'mp2', '48c3']  # pooling fits to the shape, no padding is required
    with tf.Graph().as_default(), tf.Session() as ses:
        input_x = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        with tf.variable_scope('padded'):
            padding_encoded, padding_decoded = \
              eltf.models.cnn_autoencoder(input_x, padding_encoder, use_bn=False)
        with tf.variable_scope('fitted'):
            fitted_encoded, fitted_decoded = \
              eltf.models.cnn_autoencoder(input_x, fitted_encoder, use_bn=False)
        ses.run(tf.local_variables_initializer())
        ses.run(tf.global_variables_initializer())
        # test padding encoder
        padding_encoded_value = padding_encoded.eval(feed_dict={input_x: np.ones(input_shape)}, session=ses)
        assert padding_encoded_value.shape == padding_encoded_shape
        padding_decoded_value = padding_decoded.eval(feed_dict={input_x: np.ones(input_shape)}, session=ses)
        assert padding_decoded_value.shape == input_shape
        # test fitted encoder
        fitted_encoded_value = fitted_encoded.eval(feed_dict={input_x: np.ones(input_shape)}, session=ses)
        assert fitted_encoded_value.shape == fitted_encoded_shape
        fitted_decoded_value = fitted_decoded.eval(feed_dict={input_x: np.ones(input_shape)}, session=ses)
        assert fitted_decoded_value.shape == input_shape


def test_model_integration():
    """Test if cnn (auto)encoder is well integrated with eltf BaseModel."""
    model = SimpleAutoencoder(inputs=['images', 'masks'], outputs=['loss', 'probabilities'],
                              dataset=None, optimizer=_OPTIMIZER, log_dir=None)
    outputs = model.run({'images': np.ones((3, 100, 100, 3)), 'masks': np.zeros((3, 100, 100), dtype=np.uint8)},
                        train=False)
    assert outputs['probabilities'].shape == (3, 100, 100)

    # test spanning across multiple GPUs
    multi_gpu_model = SimpleAutoencoder(inputs=['images', 'masks'], outputs=['loss', 'probabilities'],
                                        dataset=None, optimizer=_OPTIMIZER, log_dir=None, n_gpus=4,
                                        session_config={'allow_soft_placement': True})
    outputs2 = multi_gpu_model.run({'images': np.ones((3, 100, 100, 3)),
                                    'masks': np.zeros((3, 100, 100), dtype=np.uint8)}, train=True)
    assert outputs2['probabilities'].shape == (3, 100, 100)
