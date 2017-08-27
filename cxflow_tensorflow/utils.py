"""
Module with various TF related helper functions
"""

from typing import Dict, Callable, Any

import numpy as np
import tensorflow as tf

from cxflow.utils.reflection import parse_fully_qualified_name, create_object, get_class_module, get_attribute


TF_OPTIMIZERS_MODULE = 'tensorflow.python.training'

TF_ACTIVATIONS_MODULE = 'tensorflow.python.ops.nn'
"""Module with TensorFlow activation functions."""


def create_optimizer(optimizer_config: Dict[str, Any]):
    """
    Create tf optimizer according to the given config.

    When `module` entry is not present in the optimizer_config,
    the function attempts to find it under the TF_OPTIMIZER_MODULE.

    A tf variable named 'learning_rate' is created during the process.
    One must handle Graphs and Sessions carefully when using this function.

    :param optimizer_config: dict with at least `class` and `learning_rate` entries
    :return: optimizer
    """
    assert 'learning_rate' in optimizer_config, 'Optimizer learning rate not specified'
    assert 'class' in optimizer_config, 'Optimizer class not specified'

    optimizer_module, optimizer_class = parse_fully_qualified_name(optimizer_config['class'])
    if optimizer_module is None:
        optimizer_module = get_class_module(TF_OPTIMIZERS_MODULE, optimizer_class)

    kwargs = optimizer_config.copy()
    kwargs.pop('class')

    learning_rate = kwargs.pop('learning_rate')
    learning_rate = tf.Variable(learning_rate, name='learning_rate', trainable=False)

    return create_object(optimizer_module, optimizer_class, args=(learning_rate,), kwargs=kwargs)


def create_activation(activation_name: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Create TensorFlow activation function with the given name.

    List of available activation functions is available in
    `TensorFlow docs <https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_>`_.

    :param activation_name: activation function name
    :return: callable activation function
    """
    if activation_name == 'identity':
        return tf.identity
    return get_attribute(TF_ACTIVATIONS_MODULE, activation_name)


def repeat(tensor: tf.Tensor, repeats: int, axis: int) -> tf.Tensor:
    """
    Repeat elements of the input tensor in the specified axis repeat-times.
    """
    shape = tensor.get_shape().as_list()

    dims = np.arange(len(tensor.shape))
    prepare_perm = np.hstack(([axis], np.delete(dims, axis)))
    restore_perm = np.hstack((dims[1:axis+1], [0], dims[axis+1:]))

    indices = tf.cast(tf.floor(tf.range(0, shape[axis]*repeats)/tf.constant(repeats)), 'int32')

    shuffled = tf.transpose(tensor, prepare_perm)
    repeated = tf.gather(shuffled, indices)
    return tf.transpose(repeated, restore_perm)
