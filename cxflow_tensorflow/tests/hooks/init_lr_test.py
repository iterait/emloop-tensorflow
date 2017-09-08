"""
Test module for cxflow_tensorflow.hooks.InitLR hook.
"""
from unittest import TestCase
from cxflow_tensorflow import InitLR
from .decay_lr_test import LRModel


class InitLRTest(TestCase):
    """
    Test case for ``InitLR`` hook.
    """

    def test_creation(self):
        """ Test ``InitLR.__init__`` raises when configured improperly."""
        model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
        self.assertRaises(TypeError, InitLR, model=42, value=0.1)
        self.assertRaises(KeyError, InitLR, model=model, variable_name='does_not_exist', value=0.1)

    def test_init(self):
        """Test ``InitLR`` hook initializes the variable properly."""
        init_value = 0.001

        model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
        hook = InitLR(model, value=init_value)

        lr = model.graph.get_tensor_by_name('learning_rate:0')
        self.assertAlmostEqual(lr.eval(session=model.session), 2, places=3)
        hook.before_training()
        self.assertAlmostEqual(lr.eval(session=model.session), init_value, places=3)
