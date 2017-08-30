"""
Test module for cxflow_tensorflow.hooks.LRDecayHook.
"""

import tensorflow as tf
from unittest import TestCase

from cxflow_tensorflow import DecayLR

from ..model_test import TrainableModel, _OPTIMIZER


class LRModel(TrainableModel):

    def _create_model(self, **kwargs):

        tf.Variable(2, name='learning_rate', dtype=tf.float32)

        super()._create_model(**kwargs)

    def _create_train_ops(self, _):
        tf.no_op(name='train_op_1')


class LRDecayHookTest(TestCase):
    """
    Test case for LRDecayHook.
    """

    def test_invalid_config(self):
        """ Test ``LRDecayHook`` invalid configurations."""

        model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])

        self.assertRaises(TypeError, DecayLR, model=42)
        self.assertRaises(ValueError, DecayLR, model, decay_value=-1)
        self.assertRaises(ValueError, DecayLR, model, decay_type='unrecognized')
        self.assertRaises(KeyError, DecayLR, model, variable_name='missing_variable')

    def test_multiply(self):
        """ Test if ``LRDecayHook`` works properly in multiply mode."""

        decay_value = 0.9
        repeats = 13

        model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
        hook = DecayLR(model, decay_value=decay_value)

        hook.after_epoch()
        self.assertAlmostEqual(
            model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session), 2*decay_value)

        for _ in range(repeats):
            hook.after_epoch()
        self.assertAlmostEqual(model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session),
                               2*(decay_value**(1+repeats)))

    def test_add(self):
        """ Test if ``LRDecayHook`` works properly in addition mode."""
        decay_value = 0.01
        repeats = 17

        model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
        hook = DecayLR(model, decay_value=decay_value, decay_type='add')

        hook.after_epoch()
        self.assertAlmostEqual(model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session),
                               2+decay_value, places=3)

        for _ in range(repeats):
            hook.after_epoch()
        self.assertAlmostEqual(model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session),
                               2+(decay_value*(1+repeats)), places=3)
