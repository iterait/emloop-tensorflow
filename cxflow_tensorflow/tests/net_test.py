"""
Test module for base tensorflow nets (cxflow.nets.net).
"""
import os
from os import path
import tempfile
import shutil

import numpy as np
import tensorflow as tf

from cxflow import MainLoop
from cxflow.tests.main_loop_test import SimpleDataset
from cxflow.hooks import EpochStopperHook

from cxflow_tensorflow import BaseNet, create_optimizer
from cxflow_tensorflow.tests.test_core import CXTestCaseWithDirAndNet


def create_simple_main_loop(epochs: int, tmpdir: str):
    dataset = SimpleDataset()
    net = TrainableNet(dataset=dataset, log_dir=tmpdir,  # pylint: disable=redefined-variable-type
                       inputs=['input', 'target'], outputs=['output'])
    mainloop = MainLoop(net=net, dataset=dataset, hooks=[EpochStopperHook(epoch_limit=epochs)],
                        skip_zeroth_epoch=False)
    return dataset, net, mainloop


class DummyNet(BaseNet):
    """Dummy tf net with empty graph."""

    def _create_net(self, **kwargs):
        """Create dummy tf net."""

        # defining dummy variable as otherwise we would not be able to create the net saver
        tf.Variable(name='dummy', initial_value=[1])

        # defining a dummy train_op as otherwise we would not be able to create the net
        tf.no_op(name='train_op')


class TrainOpNet(BaseNet):
    """Dummy tf net with train op saved in self."""

    def _create_net(self, **kwargs):
        """Create dummy tf net."""

        # defining dummy variable as otherwise we would not be able to create the net saver
        tf.Variable(name='dummy', initial_value=[1])

        # defining a dummy train_op as otherwise we would not be able to create the net
        self.defined_train_op = tf.no_op(name='train_op')


class NoTrainOpNet(BaseNet):
    """Dummy tf net without train op."""

    def _create_net(self, **kwargs):
        """Create dummy tf net."""

        # defining dummy variable as otherwise we would not be able to create the net saver
        tf.Variable(name='dummy', initial_value=[1])

        # defining an op that is not named `train_op`
        tf.no_op(name='not_a_train_op')


class SimpleNet(BaseNet):
    """Simple net with input and output tensors."""

    def _create_net(self, **kwargs):
        """Create simple tf net."""

        self.input1 = tf.placeholder(tf.int32, shape=[None, 10], name='input')
        self.input2 = tf.placeholder(tf.int32, shape=[None, 10], name='second_input')

        self.const = tf.Variable([2]*10, name='const')

        self.output = tf.multiply(self.input1, self.const, name='output')

        self.sum = tf.add(self.input1, self.input2, name='sum')

        # defining a dummy train_op as otherwise we would not be able to create the net
        self.defined_train_op = tf.no_op(name='train_op')

        self.session.run(tf.global_variables_initializer())


class TrainableNet(BaseNet):
    """Trainable tf net."""

    def _create_net(self, **kwargs):
        """Create simple trainable tf net."""

        self.input = tf.placeholder(tf.float32, shape=[None, 10], name='input')
        self.target = tf.placeholder(tf.float32, shape=[None, 10], name='target')

        self.var = tf.Variable([2] * 10, name='var', dtype=tf.float32)

        self.output = tf.multiply(self.input, self.var, name='output')

        loss = tf.reduce_mean(tf.squared_difference(self.target, self.output))

        # defining a dummy train_op as otherwise we would not be able to create the net
        create_optimizer({'module': 'tensorflow.python.training.adam',
                          'class': 'AdamOptimizer', 'learning_rate': 0.1}).minimize(loss, name='train_op')

        self.session.run(tf.global_variables_initializer())


class BaseNetTest(CXTestCaseWithDirAndNet):
    """
    Test case for BaseNet.

    Note: do not forget to reset the default graph after every net creation!
    """

    def test_init_asserts(self):
        """Test if the init arguments are correctly asserted."""

        good_io = {'inputs': [], 'outputs': ['dummy']}
        DummyNet(dataset=None, log_dir='', **good_io)
        tf.reset_default_graph()

        # test assertion on missing in/out
        self.assertRaises(AssertionError, DummyNet, dataset=None, log_dir='', inputs=['a'], outputs=[])
        tf.reset_default_graph()

        # test assertion on negative thread count
        DummyNet(dataset=None, log_dir='', threads=2, **good_io)
        self.assertRaises(AssertionError, DummyNet, dataset=None, log_dir='', threads=-2, **good_io)
        tf.reset_default_graph()

    def test_finding_train_op(self):
        """Test finding train op in graph."""

        good_io = {'inputs': [], 'outputs': ['dummy']}

        # test whether train_op is found correctly
        trainop_net = TrainOpNet(dataset=None, log_dir='', **good_io)
        self.assertEqual(trainop_net.defined_train_op, trainop_net.train_op)
        tf.reset_default_graph()

        # test whether an error is raised when no train_op is defined
        self.assertRaises(ValueError, NoTrainOpNet, dataset=None, log_dir='', **good_io)
        tf.reset_default_graph()

    def test_io_mapping(self):
        """Test if net.io is translated to output/input names."""

        good_io = {'inputs': ['input', 'second_input'], 'outputs': ['output', 'sum']}
        net = SimpleNet(dataset=None, log_dir='', **good_io)
        self.assertListEqual(net.input_names, good_io['inputs'])
        self.assertListEqual(net.output_names, good_io['outputs'])
        tf.reset_default_graph()

        # test if an error is raised when certain input/output tensor is not found
        self.assertRaises(ValueError, SimpleNet, dataset=None, log_dir='',
                          inputs=['input', 'second_input', 'third_input'], outputs=['output', 'sum'])
        tf.reset_default_graph()
        self.assertRaises(ValueError, SimpleNet, dataset=None, log_dir='', inputs=['input', 'second_input'],
                          outputs=['output', 'sum', 'sub'])
        tf.reset_default_graph()

    def test_get_tensor_by_name(self):
        """Test if _get_tensor_by_name works properly."""

        net = SimpleNet(dataset=None, log_dir='', inputs=['input', 'second_input'], outputs=['output', 'sum'])
        self.assertEqual(net.get_tensor_by_name('sum'), net.sum)
        self.assertRaises(KeyError, net.get_tensor_by_name, name='not_in_graph')
        tf.reset_default_graph()

    def test_run(self):
        """Test tf net run."""

        good_io = {'inputs': ['input', 'second_input'], 'outputs': ['output', 'sum']}
        net = SimpleNet(dataset=None, log_dir='', **good_io)
        valid_batch = {'input': [[1]*10], 'second_input': [[2]*10]}

        # test if outputs are correctly returned
        results = net.run(batch=valid_batch, train=False)
        for output_name in good_io['outputs']:
            self.assertTrue(output_name in results)
        self.assertTrue(np.allclose(results['output'], [2]*10))
        self.assertTrue(np.allclose(results['sum'], [3]*10))
        tf.reset_default_graph()

        # test variables update if and only if train=True
        trainable_net = TrainableNet(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
        batch = {'input': [[1]*10], 'target': [[0]*10]}

        # single run with train=False
        orig_value = trainable_net.var.eval(session=trainable_net.session)
        trainable_net.run(batch, train=False)
        after_value = trainable_net.var.eval(session=trainable_net.session)
        self.assertTrue(np.allclose(orig_value, after_value))

        # multiple runs with train=False
        for _ in range(100):
            trainable_net.run(batch, train=False)
        after_value = trainable_net.var.eval(session=trainable_net.session)
        self.assertTrue(np.allclose(orig_value, after_value))

        # single run with train=True
        trainable_net.run(batch, train=True)
        after_value = trainable_net.var.eval(session=trainable_net.session)
        self.assertFalse(np.allclose(orig_value, after_value))

        # multiple runs with train=True
        trainable_net.run(batch, train=True)
        for _ in range(1000):
            trainable_net.run(batch, train=True)
        after_value = trainable_net.var.eval(session=trainable_net.session)
        self.assertTrue(np.allclose([0]*10, after_value))

    def test_mainloop_net_training(self):
        """Test the net is being trained properly."""
        _, net, mainloop = create_simple_main_loop(130, self.tmpdir)
        mainloop.run_training()
        after_value = net.graph.get_tensor_by_name('var:0').eval(session=net.session)
        self.assertTrue(np.allclose([0]*10, after_value, atol=0.01))

    def test_mainloop_zero_epoch_not_training(self):
        """Test the net is not being trained in the zeroth epoch."""
        _, net, mainloop = create_simple_main_loop(0, self.tmpdir)
        mainloop.run_training()
        after_value = net.graph.get_tensor_by_name('var:0').eval(session=net.session)
        self.assertTrue(np.allclose([2]*10, after_value, atol=0.01))

    def test_restore_1(self):
        """Test restore from directory with one valid checkpoint."""

        # test net saving
        trainable_io = {'inputs': ['input', 'target'], 'outputs': ['output']}
        trainable_net = TrainableNet(dataset=None, log_dir=self.tmpdir, **trainable_io)
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            trainable_net.run(batch, train=True)
        saved_var_value = trainable_net.var.eval(session=trainable_net.session)
        trainable_net.save('1')

        tf.reset_default_graph()

        # test restoring
        restored_net = BaseNet(dataset=None, log_dir='', restore_from=self.tmpdir, **trainable_io)

        var = restored_net.graph.get_tensor_by_name('var:0')
        var_value = var.eval(session=restored_net.session)
        self.assertTrue(np.allclose(saved_var_value, var_value))

    def test_restore_2_with_spec(self):
        """Test restore from directory with two checkpoints and a specification of which one to restore from."""

        # test net saving
        trainable_io = {'inputs': ['input', 'target'], 'outputs': ['output']}
        trainable_net = TrainableNet(dataset=None, log_dir=self.tmpdir, **trainable_io)
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            trainable_net.run(batch, train=True)
        saved_var_value = trainable_net.var.eval(session=trainable_net.session)
        trainable_net.save('1')
        checkpoint_path = trainable_net.save('2')

        tf.reset_default_graph()

        # test restoring
        restored_net = BaseNet(dataset=None, log_dir='', restore_from=self.tmpdir,
                               restore_model_name=checkpoint_path, **trainable_io)

        var = restored_net.graph.get_tensor_by_name('var:0')
        var_value = var.eval(session=restored_net.session)
        self.assertTrue(np.allclose(saved_var_value, var_value))

    def test_restore_2_without_spec(self):
        """Test restore from directory with two checkpoints and no specification of which one to restore from."""

        # test net saving
        trainable_io = {'inputs': ['input', 'target'], 'outputs': ['output']}
        trainable_net = TrainableNet(dataset=None, log_dir=self.tmpdir, **trainable_io)
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            trainable_net.run(batch, train=True)
        trainable_net.save('1')
        trainable_net.save('2')

        tf.reset_default_graph()

        # test restoring
        with self.assertRaises(ValueError):
            BaseNet(dataset=None, log_dir='', restore_from=self.tmpdir, **trainable_io)

    def test_restore_0(self):
        """Test restore from directory with no checkpoints."""

        # test net saving
        trainable_io = {'inputs': ['input', 'target'], 'outputs': ['output']}
        trainable_net = TrainableNet(dataset=None, log_dir=self.tmpdir, **trainable_io)
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            trainable_net.run(batch, train=True)

        tf.reset_default_graph()

        # test restoring
        with self.assertRaises(ValueError):
            BaseNet(dataset=None, log_dir='', restore_from=self.tmpdir, **trainable_io)

    def test_restore_and_train(self):
        """Test net training after restoring."""

        # save a net that is not trained
        trainable_io = {'inputs': ['input', 'target'], 'outputs': ['output']}
        trainable_net = TrainableNet(dataset=None, log_dir=self.tmpdir, **trainable_io)
        trainable_net.save('')
        tf.reset_default_graph()

        # restored the net
        restored_net = BaseNet(dataset=None, log_dir='', restore_from=self.tmpdir, **trainable_io)

        # test whether it can be trained
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            restored_net.run(batch, train=True)

        after_value = restored_net.graph.get_tensor_by_name('var:0').eval(session=restored_net.session)
        self.assertTrue(np.allclose([0]*10, after_value))


class TFBaseNetSaverTest(CXTestCaseWithDirAndNet):
    """
    Test case for correct usage of tensorflow saver in BaseNet.
    """

    def test_keep_checkpoints(self):
        """
        Test if the checkpoints are kept.

        This is regression test for issue #71 (tensorflow saver is keeping only the last 5 checkpoints).
        """
        dummy_net = SimpleNet(dataset=None, log_dir=self.tmpdir, inputs=[], outputs=['output'])

        checkpoints = []
        for i in range(20):
            checkpoints.append(dummy_net.save(str(i)))

        for checkpoint in checkpoints:
            self.assertTrue(path.exists(checkpoint+'.index'))
            self.assertTrue(path.exists(checkpoint+'.meta'))
            data_prefix = path.basename(checkpoint)+'.data'
            data_files = [file for file in os.listdir(path.dirname(checkpoint)) if file.startswith(data_prefix)]
            self.assertGreater(len(data_files), 0)


class TFBaseNetManagementTest(CXTestCaseWithDirAndNet):
    """
    Test case for correct management of tf graphs and sessions.
    """

    def test_two_nets_created(self):
        """
        Test if one can create and train two BaseNets.

        This is regression test for issue #83 (One can not create and use more than one instance of BaseNet).
        """

        trainable_io = {'inputs': ['input', 'target'], 'outputs': ['output']}
        net1 = TrainableNet(dataset=None, log_dir='', **trainable_io)
        net2 = TrainableNet(dataset=None, log_dir='', **trainable_io)
        batch = {'input': [[1]*10], 'target': [[0]*10]}

        # test if one can train one net while the other remains intact
        for _ in range(1000):
            net1.run(batch, train=True)
        trained_value = net1.var.eval(session=net1.session)
        self.assertTrue(np.allclose([0]*10, trained_value))
        default_value = net2.var.eval(session=net2.session)
        self.assertTrue(np.allclose([2]*10, default_value))

        # test if one can train the other net
        for _ in range(1000):
            net2.run(batch, train=True)
        trained_value2 = net2.var.eval(session=net2.session)
        self.assertTrue(np.allclose([0] * 10, trained_value2))

    def test_two_nets_restored(self):
        """
        Test if one can `_restore_network` and use two BaseNets.

        This is regression test for issue #83 (One can not create and use more than one instance of BaseNet).
        """
        tmpdir2 = tempfile.mkdtemp()

        trainable_io = {'inputs': ['input', 'target'], 'outputs': ['output']}
        net1 = TrainableNet(dataset=None, log_dir=self.tmpdir, **trainable_io)
        net2 = TrainableNet(dataset=None, log_dir=tmpdir2, **trainable_io)
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            net1.run(batch, train=True)

        checkpoint_path1 = net1.save('')
        checkpoint_path2 = net2.save('')

        # test if one can `_restore_network` two nets and use them at the same time
        restored_net1 = BaseNet(dataset=None, log_dir='', restore_from=self.tmpdir, **trainable_io)
        restored_net2 = BaseNet(dataset=None, log_dir='', restore_from=tmpdir2, **trainable_io)

        trained_value = restored_net1.graph.get_tensor_by_name('var:0').eval(session=restored_net1.session)
        self.assertTrue(np.allclose([0]*10, trained_value))
        default_value = restored_net2.graph.get_tensor_by_name('var:0').eval(session=restored_net2.session)
        self.assertTrue(np.allclose([2]*10, default_value))

        shutil.rmtree(tmpdir2)
