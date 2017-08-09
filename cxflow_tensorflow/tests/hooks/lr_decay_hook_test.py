"""
Test module for cxflow_tensorflow.hooks.LRDecayHook.
"""

import tensorflow as tf
from unittest import TestCase

from cxflow_tensorflow import LRDecayHook

from ..tf_net_test import TrainableNet, SimpleDataset, EpochStopperHook, MainLoop


class LRNet(TrainableNet):

    def _create_net(self, **kwargs):

        tf.Variable(2, name='learning_rate', dtype=tf.float32)

        super()._create_net(**kwargs)


class LRDecayHookTest(TestCase):
    """
    Test case for LRDecayHook.
    """

    def test_invalid_config(self):
        """ Test LRDecayHook invalid configurations."""

        net = LRNet(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])

        self.assertRaises(TypeError, LRDecayHook, net=42)
        self.assertRaises(ValueError, LRDecayHook, net, decay_value=-1)
        self.assertRaises(ValueError, LRDecayHook, net, decay_type='unrecognized')
        self.assertRaises(KeyError, LRDecayHook, net, variable_name='missing_variable')

    def test_multiply(self):
        """ Test if LRDecayHook works properly in multiply mode."""

        decay_value = 0.9
        repeats = 13

        net = LRNet(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
        hook = LRDecayHook(net, decay_value=decay_value)

        hook.after_epoch()
        self.assertAlmostEqual(net.graph.get_tensor_by_name('learning_rate:0').eval(session=net.session), 2*decay_value)

        for _ in range(repeats):
            hook.after_epoch()
        self.assertAlmostEqual(net.graph.get_tensor_by_name('learning_rate:0').eval(session=net.session),
                               2*(decay_value**(1+repeats)))

    def test_add(self):
        """ Test if LRDecayHook works properly in addition mode."""
        decay_value = 0.01
        repeats = 17

        net = LRNet(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
        hook = LRDecayHook(net, decay_value=decay_value, decay_type='add')

        hook.after_epoch()
        self.assertAlmostEqual(net.graph.get_tensor_by_name('learning_rate:0').eval(session=net.session),
                               2+decay_value, places=3)

        for _ in range(repeats):
            hook.after_epoch()
        self.assertAlmostEqual(net.graph.get_tensor_by_name('learning_rate:0').eval(session=net.session),
                               2+(decay_value*(1+repeats)), places=3)
