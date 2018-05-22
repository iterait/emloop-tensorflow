"""
Test module for cxflow_tensorflow.hooks.DecayLR hook.
"""
import unittest.mock as mock
from unittest import TestCase
import tensorflow as tf

from cxflow_tensorflow.hooks import DecayLR, DecayLROnPlateau
from cxflow.hooks import OnPlateau

from ..model_test import TrainableModel


class LRModel(TrainableModel):

    def _create_model(self, **kwargs):

        tf.Variable(2, name='learning_rate', dtype=tf.float32)

        super()._create_model(**kwargs)

    def _create_train_ops(self, *_):
        tf.no_op(name='train_op_1')


class DecayLRTest(TestCase):
    """
    Test case for DecayLR.
    """

    def test_invalid_config(self):
        """ Test ``DecayLR`` invalid configurations."""

        model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])

        self.assertRaises(TypeError, DecayLR, model=42)
        self.assertRaises(ValueError, DecayLR, model, decay_value=-1)
        self.assertRaises(ValueError, DecayLR, model, decay_type='unrecognized')
        self.assertRaises(KeyError, DecayLR, model, variable_name='missing_variable')

    def test_multiply(self):
        """ Test if ``DecayLR`` works properly in multiply mode."""

        decay_value = 0.9
        repeats = 13

        model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
        hook = DecayLR(model, decay_value=decay_value)

        hook._after_n_epoch(1)
        self.assertAlmostEqual(
            model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session), 2*decay_value)

        for _ in range(repeats):
            hook._after_n_epoch(1)
        self.assertAlmostEqual(model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session),
                               2*(decay_value**(1+repeats)))

    def test_multiply_n_epoch(self):
        """ Test if ``DecayLR`` works properly in multiply mode for every n epoch."""    

        decay_value = 0.9
        repeats = 13
        n = 2

        for _ in range(repeats):
            hook._after_n_epoch(n)
        self.assertAlmostEqual(model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session),
                               2*(decay_value**(1+(repeats//n))))

    def test_add(self):
        """ Test if ``DecayLR`` works properly in addition mode."""

        decay_value = 0.01
        repeats = 17

        model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
        hook = DecayLR(model, decay_value=decay_value, decay_type='add')

        hook._after_n_epoch(1)
        self.assertAlmostEqual(model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session),
                               2+decay_value, places=3)

        for _ in range(repeats):
            hook._after_n_epoch(1)
        self.assertAlmostEqual(model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session),
                               2+(decay_value*(1+repeats)), places=3)

    def test_add_n_epoch(self):
        """ Test if ``DecayLR`` works properly in addition mode for every n epoch."""

        decay_value = 0.01
        repeats = 17
        n = 2

        for _ in range(repeats):
            hook._after_n_epoch(n)
        self.assertAlmostEqual(model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session),
                               2+(decay_value*(1+(repeats//n))), places=3)


class DecayLROnPlateauTest(TestCase):
    """
    Test case for DecayLROnPlateau.

    Both OnPlateau and DecayLR are already tested, we only need to check if they are properly integrated.
    """

    def get_model(self):
        return LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])

    def get_epoch_data(self):
        return {'valid': {'loss': [0]}}

    def test_arg_forward(self):
        """ Test if ``DecayLROnPlateau`` forwards args properly."""
        hook = DecayLROnPlateau(model=self.get_model(), short_term=5, variable='my_lr')
        self.assertEqual(hook._variable, 'my_lr')
        self.assertEqual(hook._short_term, 5)

    def test_call_forward(self):
        """ Test if ``DecayLROnPlateau`` forwards event calls properly."""
        with mock.patch.object(DecayLR, 'after_epoch') as decay_ae, \
             mock.patch.object(OnPlateau, 'after_epoch') as plateau_ae:
            hook = DecayLROnPlateau(model=self.get_model())
            hook.after_epoch(epoch_id=0, epoch_data=self.get_epoch_data())
            self.assertEqual(decay_ae.call_count, 0)
            self.assertEqual(plateau_ae.call_count, 1)

    def test_wait(self):
        """ Test if ``DecayLROnPlateau`` waits short_term epochs between decays."""
        with mock.patch.object(DecayLR, '_decay_variable') as decay:
            hook = DecayLROnPlateau(model=self.get_model(), long_term=4, short_term=3)
            hook._on_plateau_action()
            for i in range(hook._long_term):
                self.assertEqual(decay.call_count, 1)
                hook.after_epoch(epoch_id=i, epoch_data=self.get_epoch_data())
                hook._on_plateau_action()
            self.assertEqual(decay.call_count, 2)
