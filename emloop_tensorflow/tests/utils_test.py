"""
Test module for utils module.
"""

import tensorflow as tf
import pytest

from emloop_tensorflow import create_optimizer, create_activation


def test_create_optimizer():
    """Test if create optimizer does work with tf optimizers."""
    optimizer_config = {'learning_rate': 0.1}

    # test missing required entry `class`
    with pytest.raises(AssertionError):
        create_optimizer(optimizer_config)

    optimizer_config['class'] = 'tensorflow.python.training.gradient_descent.GradientDescentOptimizer'

    with tf.Session().as_default():
        # test if the optimizer is created correctlyW
        optimizer = create_optimizer(optimizer_config)
        assert isinstance(optimizer, tf.train.GradientDescentOptimizer)

        # test if learning_rate variable is created with the correct value
        lr_tensor = tf.get_default_graph().get_tensor_by_name('learning_rate:0')
        tf.get_default_session().run(tf.global_variables_initializer())
        assert round(abs(lr_tensor.eval()-0.1), 7) == 0

    optimizer_config2 = {'learning_rate': 0.1, 'class': 'tensorflow.python.training.momentum.MomentumOptimizer'}

    # test missing required argument (momentum in this case)
    with tf.Graph().as_default():
        with pytest.raises(TypeError):
            create_optimizer(optimizer_config2)


def test_create_activation():
    """Test if create activation works properly."""
    assert create_activation('relu') is tf.nn.relu
    assert create_activation('identity') is tf.identity
    with pytest.raises(AttributeError):
        create_activation('i_do_not_exist')
