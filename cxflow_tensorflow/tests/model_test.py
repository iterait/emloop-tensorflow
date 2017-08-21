"""
Test module for base tensorflow models (cxflow.models.model).
"""
import os
from os import path
import tempfile
import shutil

import numpy as np
import tensorflow as tf

from cxflow import MainLoop
from cxflow.tests.main_loop_test import SimpleDataset
from cxflow.tests.test_core import CXTestCaseWithDir
from cxflow.hooks import StopAfter

from cxflow_tensorflow import BaseModel

_OPTIMIZER = {'module': 'tensorflow.python.training.adam', 'class': 'AdamOptimizer', 'learning_rate': 0.1}
_IO = {'inputs': ['input', 'target'], 'outputs': ['output', 'loss']}


def create_simple_main_loop(epochs: int, tmpdir: str):
    dataset = SimpleDataset()
    model = TrainableModel(dataset=dataset, log_dir=tmpdir, **_IO, optimizer=_OPTIMIZER)
    mainloop = MainLoop(model=model, dataset=dataset, hooks=[StopAfter(epochs=epochs)], skip_zeroth_epoch=False)
    return dataset, model, mainloop


class DummyModel(BaseModel):
    """Dummy tf model with empty graph."""

    def _create_model(self, **kwargs):
        """Create dummy tf model."""

        # defining dummy variable as otherwise we would not be able to create the model saver
        tf.Variable(name='dummy', initial_value=[1])

    def _create_train_ops(self, _):
        tf.no_op(name='train_op_1')


class TrainOpModel(BaseModel):
    """Dummy tf model with train op saved in self."""

    def _create_model(self, **kwargs):
        """Create dummy tf model."""

        # defining dummy variable as otherwise we would not be able to create the model saver
        tf.Variable(name='dummy', initial_value=[1])

    def _create_train_ops(self, _):
        tf.no_op(name='train_op_1')


class NoTrainOpModel(BaseModel):
    """Dummy tf model without train op."""

    def _create_model(self, **kwargs):
        """Create dummy tf model."""

        # defining dummy variable as otherwise we would not be able to create the model saver
        tf.Variable(name='dummy', initial_value=[1])

    def _create_train_ops(self, _):
        pass


class SimpleModel(BaseModel):
    """Simple model with input and output tensors."""

    def _create_model(self, **kwargs):
        """Create simple tf model."""

        self.input1 = tf.placeholder(tf.int32, shape=[None, 10], name='input')
        self.input2 = tf.placeholder(tf.int32, shape=[None, 10], name='second_input')

        self.const = tf.Variable([2]*10, name='const')

        self.output = tf.multiply(self.input1, self.const, name='output')

        self.sum = tf.add(self.input1, self.input2, name='sum')

    def _create_train_ops(self, _):
        tf.no_op(name='train_op_1')


class TrainableModel(BaseModel):
    """Trainable tf model."""

    def _create_model(self, **kwargs):
        """Create simple trainable tf model."""

        self.input = tf.placeholder(tf.float32, shape=[None, 10], name='input')
        self.target = tf.placeholder(tf.float32, shape=[None, 10], name='target')

        self.var = tf.get_variable(name='var', shape=[10], dtype=tf.float32,
                                   initializer=tf.constant_initializer([2] * 10))

        self.output = tf.multiply(self.input, self.var, name='output')

        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.output), axis=-1)
        tf.identity(self.loss, name='loss')


class BaseModelTest(CXTestCaseWithDir):
    """
    Test case for BaseModel.

    Note: do not forget to reset the default graph after every model creation!
    """

    def test_finding_train_op(self):
        """Test finding train op in graph."""

        good_io = {'inputs': [], 'outputs': ['dummy']}

        # test whether train_op is found correctly
        TrainOpModel(dataset=None, log_dir='', **good_io)

        # test whether an error is raised when no train_op is defined
        self.assertRaises(ValueError, NoTrainOpModel, dataset=None, log_dir='', **good_io)

    def test_io_mapping(self):
        """Test if model.io is translated to output/input names."""

        good_io = {'inputs': ['input', 'second_input'], 'outputs': ['output', 'sum']}
        model = SimpleModel(dataset=None, log_dir='', **good_io)
        self.assertListEqual(model.input_names, good_io['inputs'])
        self.assertListEqual(model.output_names, good_io['outputs'])

        # test if an error is raised when certain input/output tensor is not found
        self.assertRaises(ValueError, SimpleModel, dataset=None, log_dir='',
                          inputs=['input', 'second_input', 'third_input'], outputs=['output', 'sum'])

        self.assertRaises(ValueError, SimpleModel, dataset=None, log_dir='', inputs=['input', 'second_input'],
                          outputs=['output', 'sum', 'sub'])

    def test_run(self):
        """Test tf model run."""
        good_io = {'inputs': ['input', 'second_input'], 'outputs': ['output', 'sum']}
        model = SimpleModel(dataset=None, log_dir='', **good_io)
        valid_batch = {'input': [[1]*10], 'second_input': [[2]*10]}

        # test if outputs are correctly returned
        results = model.run(batch=valid_batch, train=False)
        for output_name in good_io['outputs']:
            self.assertTrue(output_name in results)
        self.assertTrue(np.allclose(results['output'], [2]*10))
        self.assertTrue(np.allclose(results['sum'], [3]*10))

        # test variables update if and only if train=True
        trainable_model = TrainableModel(dataset=None, log_dir='', **_IO, optimizer=_OPTIMIZER)
        batch = {'input': [[1]*10], 'target': [[0]*10]}

        # single run with train=False
        orig_value = trainable_model.var.eval(session=trainable_model.session)
        trainable_model.run(batch, train=False)
        after_value = trainable_model.var.eval(session=trainable_model.session)
        self.assertTrue(np.allclose(orig_value, after_value))

        # multiple runs with train=False
        for _ in range(100):
            trainable_model.run(batch, train=False)
        after_value = trainable_model.var.eval(session=trainable_model.session)
        self.assertTrue(np.allclose(orig_value, after_value))

        # single run with train=True
        trainable_model.run(batch, train=True)
        after_value = trainable_model.var.eval(session=trainable_model.session)
        self.assertFalse(np.allclose(orig_value, after_value))

        # multiple runs with train=True
        trainable_model.run(batch, train=True)
        for _ in range(1000):
            trainable_model.run(batch, train=True)
        after_value = trainable_model.var.eval(session=trainable_model.session)
        self.assertTrue(np.allclose([0]*10, after_value))

    def test_mainloop_model_training(self):
        """Test the model is being trained properly."""
        _, model, mainloop = create_simple_main_loop(130, self.tmpdir)
        mainloop.run_training()
        after_value = model.graph.get_tensor_by_name('var:0').eval(session=model.session)
        self.assertTrue(np.allclose([0]*10, after_value, atol=0.01))

    def test_mainloop_zero_epoch_not_training(self):
        """Test the model is not being trained in the zeroth epoch."""
        _, model, mainloop = create_simple_main_loop(0, self.tmpdir)
        mainloop.run_training()
        after_value = model.graph.get_tensor_by_name('var:0').eval(session=model.session)
        self.assertTrue(np.allclose([2]*10, after_value, atol=0.01))

    def test_restore_1(self):
        """Test restore from directory with one valid checkpoint."""

        # test model saving
        trainable_model = TrainableModel(dataset=None, log_dir=self.tmpdir, **_IO, optimizer=_OPTIMIZER)
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            trainable_model.run(batch, train=True)
        saved_var_value = trainable_model.var.eval(session=trainable_model.session)
        trainable_model.save('1')

        # test restoring
        restored_model = BaseModel(dataset=None, log_dir='', restore_from=self.tmpdir, **_IO, optimizer=_OPTIMIZER)

        var = restored_model.graph.get_tensor_by_name('var:0')
        var_value = var.eval(session=restored_model.session)
        self.assertTrue(np.allclose(saved_var_value, var_value))

    def test_restore_2_with_spec(self):
        """Test restore from directory with two checkpoints and a specification of which one to restore from."""

        # test model saving
        trainable_model = TrainableModel(dataset=None, log_dir=self.tmpdir, **_IO, optimizer=_OPTIMIZER)
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            trainable_model.run(batch, train=True)
        saved_var_value = trainable_model.var.eval(session=trainable_model.session)
        trainable_model.save('1')
        checkpoint_path = trainable_model.save('2')

        # test restoring
        restored_model = BaseModel(dataset=None, log_dir='', restore_from=self.tmpdir,
                                   restore_model_name=checkpoint_path, **_IO, optimizer=_OPTIMIZER)

        var = restored_model.graph.get_tensor_by_name('var:0')
        var_value = var.eval(session=restored_model.session)
        self.assertTrue(np.allclose(saved_var_value, var_value))

    def test_restore_2_without_spec(self):
        """Test restore from directory with two checkpoints and no specification of which one to restore from."""

        # test model saving
        trainable_model = TrainableModel(dataset=None, log_dir=self.tmpdir, **_IO, optimizer=_OPTIMIZER)
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            trainable_model.run(batch, train=True)
        trainable_model.save('1')
        trainable_model.save('2')

        # test restoring
        with self.assertRaises(ValueError):
            BaseModel(dataset=None, log_dir='', restore_from=self.tmpdir, **_IO)

    def test_restore_0(self):
        """Test restore from directory with no checkpoints."""

        # test model saving
        trainable_model = TrainableModel(dataset=None, log_dir=self.tmpdir, **_IO, optimizer=_OPTIMIZER)
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            trainable_model.run(batch, train=True)

        # test restoring
        with self.assertRaises(ValueError):
            BaseModel(dataset=None, log_dir='', restore_from=self.tmpdir, **_IO)

    def test_restore_and_train(self):
        """Test model training after restoring."""

        # save a model that is not trained
        trainable_model = TrainableModel(dataset=None, log_dir=self.tmpdir, **_IO, optimizer=_OPTIMIZER)
        trainable_model.save('')

        # restored the model
        restored_model = BaseModel(dataset=None, log_dir='', restore_from=self.tmpdir, **_IO)

        # test whether it can be trained
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            restored_model.run(batch, train=True)

        after_value = restored_model.graph.get_tensor_by_name('var:0').eval(session=restored_model.session)
        self.assertTrue(np.allclose([0]*10, after_value))


class TFBaseModelSaverTest(CXTestCaseWithDir):
    """
    Test case for correct usage of tensorflow saver in BaseModel.
    """

    def test_keep_checkpoints(self):
        """
        Test if the checkpoints are kept.

        This is regression test for issue #71 (tensorflow saver is keeping only the last 5 checkpoints).
        """
        dummy_model = SimpleModel(dataset=None, log_dir=self.tmpdir, inputs=[], outputs=['output'])

        checkpoints = []
        for i in range(20):
            checkpoints.append(dummy_model.save(str(i)))

        for checkpoint in checkpoints:
            self.assertTrue(path.exists(checkpoint+'.index'))
            self.assertTrue(path.exists(checkpoint+'.meta'))
            data_prefix = path.basename(checkpoint)+'.data'
            data_files = [file for file in os.listdir(path.dirname(checkpoint)) if file.startswith(data_prefix)]
            self.assertGreater(len(data_files), 0)


class TFBaseModelManagementTest(CXTestCaseWithDir):
    """
    Test case for correct management of tf graphs and sessions.
    """

    def test_two_models_created(self):
        """
        Test if one can create and train two BaseModels.

        This is regression test for issue #83 (One can not create and use more than one instance of BaseModel).
        """
        model1 = TrainableModel(dataset=None, log_dir='', **_IO, optimizer=_OPTIMIZER)
        model2 = TrainableModel(dataset=None, log_dir='', **_IO, optimizer=_OPTIMIZER)
        batch = {'input': [[1]*10], 'target': [[0]*10]}

        # test if one can train one model while the other remains intact
        for _ in range(1000):
            model1.run(batch, train=True)
        trained_value = model1.var.eval(session=model1.session)
        self.assertTrue(np.allclose([0]*10, trained_value))
        default_value = model2.var.eval(session=model2.session)
        self.assertTrue(np.allclose([2]*10, default_value))

        # test if one can train the other model
        for _ in range(1000):
            model2.run(batch, train=True)
        trained_value2 = model2.var.eval(session=model2.session)
        self.assertTrue(np.allclose([0] * 10, trained_value2))

    def test_two_models_restored(self):
        """
        Test if one can `_restore_model` and use two BaseModels.

        This is regression test for issue #83 (One can not create and use more than one instance of BaseModel).
        """
        tmpdir2 = tempfile.mkdtemp()

        model1 = TrainableModel(dataset=None, log_dir=self.tmpdir, **_IO, optimizer=_OPTIMIZER)
        model2 = TrainableModel(dataset=None, log_dir=tmpdir2, **_IO, optimizer=_OPTIMIZER)
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            model1.run(batch, train=True)

        model1.save('')
        model2.save('')

        # test if one can `_restore_model` two models and use them at the same time
        restored_model1 = BaseModel(dataset=None, log_dir='', restore_from=self.tmpdir, **_IO)
        restored_model2 = BaseModel(dataset=None, log_dir='', restore_from=tmpdir2, **_IO)

        trained_value = restored_model1.graph.get_tensor_by_name('var:0').eval(session=restored_model1.session)
        self.assertTrue(np.allclose([0]*10, trained_value))
        default_value = restored_model2.graph.get_tensor_by_name('var:0').eval(session=restored_model2.session)
        self.assertTrue(np.allclose([2]*10, default_value))

        shutil.rmtree(tmpdir2)


class TFBaseModelMultiGPUTest(CXTestCaseWithDir):
    """
    Test case for correct handling of multi-gpu trainings.
    """

    def test_incomplete_batches(self):
        """Test if incomplete batches are handled properly in multi-tower env."""
        multi_gpu_model = TrainableModel(dataset=None, log_dir='', **_IO, n_gpus=4,
                                         session_config={'allow_soft_placement': True}, optimizer=_OPTIMIZER)
        batch = {'input': [[1]*10]*8, 'target': [[0]*10]*8}
        small_batch = {'input': [[1]*10]*3, 'target': [[0]*10]*3}

        # single run with full batch
        outputs = multi_gpu_model.run(batch, train=False)
        self.assertTrue(np.allclose(outputs['output'], [[2]*10]*8))

        # single run with small batch
        outputs2 = multi_gpu_model.run(small_batch, train=False)
        self.assertTrue(np.allclose(outputs2['output'], [[2]*10]*3))

        # multiple train runs with full batch
        for _ in range(1000):
            multi_gpu_model.run(batch, train=True)
        after_value = multi_gpu_model.var.eval(session=multi_gpu_model.session)
        self.assertTrue(np.allclose(after_value, [0]*10))

        multi_gpu_model2 = TrainableModel(dataset=None, log_dir='', **_IO, n_gpus=4,
                                          session_config={'allow_soft_placement': True}, optimizer=_OPTIMIZER)
        # multiple train runs with small batch
        for _ in range(1000):
            multi_gpu_model2.run(small_batch, train=True)
        after_value = multi_gpu_model2.var.eval(session=multi_gpu_model2.session)
        self.assertTrue(np.allclose(after_value, [0]*10))
