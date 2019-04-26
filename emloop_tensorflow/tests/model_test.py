"""
Test module for base TensorFlow models (:py:class:`emloop_tensorflow.BaseModel`).
"""
import os
from os import path
import tempfile
import shutil
import pytest

import numpy as np
import tensorflow as tf

from emloop import MainLoop
from emloop.tests.main_loop_test import SimpleDataset, _DATASET_SHAPE
from emloop.hooks import StopAfter

from emloop_tensorflow import BaseModel
from emloop_tensorflow.third_party.tensorflow.freeze_graph import freeze_graph

_OPTIMIZER = {'class': 'tensorflow.python.training.adam.AdamOptimizer', 'learning_rate': 0.1}
_OPTIMIZER_CLIPPING = {'class': 'tensorflow.train.GradientDescentOptimizer', 'learning_rate': 1}
_OPTIMIZER_NO_MODULE = {'class': 'AdamOptimizer', 'learning_rate': 0.1}
_IO = {'inputs': ['input', 'target'], 'outputs': ['output', 'loss']}


def create_simple_main_loop(epochs: int, tmpdir: str):
    dataset = SimpleDataset()
    model = TrainableModel(dataset=dataset, log_dir=tmpdir, **_IO, optimizer=_OPTIMIZER)
    mainloop = MainLoop(model=model, dataset=dataset, hooks=[StopAfter(epochs=epochs)], skip_zeroth_epoch=False)
    return dataset, model, mainloop


class TrainOpModel(BaseModel):
    """Dummy TF model with train op saved in self."""

    def _create_model(self, **kwargs):
        """Create dummy tf model."""
        # defining dummy variable as otherwise we would not be able to create the model saver
        tf.Variable(name='dummy', initial_value=[1])

    def _create_train_ops(self, *_):
        tf.no_op(name='train_op_1')


class NoTrainOpModel(BaseModel):
    """Dummy TF model without train op."""

    def _create_model(self, **kwargs):
        """Create dummy tf model."""
        # defining dummy variable as otherwise we would not be able to create the model saver
        tf.Variable(name='dummy', initial_value=[1])

    def _create_train_ops(self, *_):
        pass


class SimpleModel(BaseModel):
    """Simple model with input and output tensors."""

    def _create_model(self, **kwargs):
        """Create simple TF model."""
        self.input1 = tf.placeholder(tf.int32, shape=[None, 10], name='input')
        self.input2 = tf.placeholder(tf.int32, shape=[None, 10], name='second_input')

        self.const = tf.Variable([2]*10, name='const')

        self.output = tf.multiply(self.input1, self.const, name='output')

        self.sum = tf.add(self.input1, self.input2, name='sum')

    def _create_train_ops(self, *_):
        tf.no_op(name='train_op_1')


class TrainableModel(BaseModel):
    """Trainable TF model."""

    def _create_model(self, **kwargs):
        """Create simple trainable TF model."""
        self.input = tf.placeholder(tf.float32, shape=[None, 10], name='input')
        self.target = tf.placeholder(tf.float32, shape=[None, 10], name='target')

        self.var = tf.get_variable(name='var', shape=[10], dtype=tf.float32,
                                   initializer=tf.constant_initializer([2] * 10))

        self.output = tf.multiply(self.input, self.var, name='output')
        tf.constant(0, name='scalar_output')
        tf.constant([1, 2, 3], name='batched_output')

        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.output), axis=-1, name=self._loss_name)


class RegularizedModel(TrainableModel):
    """Trainable TF model with regularization loss."""

    def _create_model(self, **kwargs):
        """Create regularized trainable TF model."""
        super()._create_model(**kwargs)
        ratio = tf.placeholder(tf.float32, shape=[1], name='ratio')
        reg_term = tf.reduce_sum(self.var)
        self.graph.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, ratio[0]*reg_term)


class DetectTrainingModel(BaseModel):
    """Model with output variable depending on the training flag."""

    def _create_model(self, **kwargs):
        self.input1 = tf.placeholder(tf.int32, shape=[None, 10], name='input')
        self.const = tf.Variable([1]*10, name='const')
        self.output = tf.identity((self.const + self.input1) * tf.cast(self.is_training, tf.int32), name='output')

    def _create_train_ops(self, *_):
        tf.no_op(name='train_op_1')


##############
# Base Model #
##############
"""
Test case for ``BaseModel``.

Note: do not forget to reset the default graph after every model creation!
"""


def test_base_class():
    """Test BaseModel can not be instantiated."""
    with pytest.raises(NotImplementedError):
        BaseModel(dataset=None, log_dir='', **{'inputs': [], 'outputs': ['dummy']})


def test_missing_optimizer():
    """Test raise if the optimizer config is missing."""
    with pytest.raises(ValueError):
        TrainableModel(dataset=None, log_dir='', **_IO)


def test_finding_train_op():
    """Test finding train op in graph."""
    good_io = {'inputs': [], 'outputs': ['dummy']}

    # test whether ``train_op`` is found correctly
    TrainOpModel(dataset=None, log_dir='', **good_io)

    # test whether an error is raised when no ``train_op`` is defined
    with pytest.raises(ValueError):
        NoTrainOpModel(dataset=None, log_dir='', **good_io)


def test_io_mapping():
    """Test if ``inputs`` and ``outputs`` are translated to output/input names."""
    good_io = {'inputs': ['input', 'second_input'], 'outputs': ['output', 'sum']}
    model = SimpleModel(dataset=None, log_dir='', **good_io)
    assert model.input_names == good_io['inputs']
    assert model.output_names == good_io['outputs']

    # test if an error is raised when certain input/output tensor is not found
    with pytest.raises(ValueError):
        SimpleModel(dataset=None, log_dir='', inputs=['input', 'second_input', 'third_input'],
                    outputs=['output', 'sum'])

    with pytest.raises(ValueError):
        SimpleModel(dataset=None, log_dir='', inputs=['input', 'second_input'], outputs=['output', 'sum', 'sub'])


def test_run(mocker):
    """Test TF model run."""
    good_io = {'inputs': ['input', 'second_input'], 'outputs': ['output', 'sum']}
    model = SimpleModel(dataset=None, log_dir='', **good_io)
    valid_batch = {'input': [[1]*10], 'second_input': [[2]*10]}

    # test if outputs are correctly returned
    results = model.run(batch=valid_batch, train=False)
    for output_name in good_io['outputs']:
        assert output_name in results
    assert np.allclose(results['output'], [2]*10)
    assert np.allclose(results['sum'], [3]*10)

    # test if buffering is properly allowed
    stream_mock = mocker.MagicMock()
    results = model.run(batch=valid_batch, train=False, stream=stream_mock)
    assert stream_mock.allow_buffering.__enter__.call_count == 1
    assert stream_mock.allow_buffering.__exit__.call_count == 1

    # test variables update if and only if ``train=True``
    trainable_model = TrainableModel(dataset=None, log_dir='', **_IO, optimizer=_OPTIMIZER)
    batch = {'input': [[1]*10], 'target': [[0]*10]}

    # single run with ``train=False``
    orig_value = trainable_model.var.eval(session=trainable_model.session)
    trainable_model.run(batch, train=False)
    after_value = trainable_model.var.eval(session=trainable_model.session)
    assert np.allclose(orig_value, after_value)

    # multiple runs with ``train=False``
    for _ in range(100):
        trainable_model.run(batch, train=False)
    after_value = trainable_model.var.eval(session=trainable_model.session)
    assert np.allclose(orig_value, after_value)

    # single run with ``train=True``
    trainable_model.run(batch, train=True)
    after_value = trainable_model.var.eval(session=trainable_model.session)
    assert not np.allclose(orig_value, after_value)

    # multiple runs with ``train=True``
    trainable_model.run(batch, train=True)
    for _ in range(1000):
        trainable_model.run(batch, train=True)
    after_value = trainable_model.var.eval(session=trainable_model.session)
    assert np.allclose([0]*10, after_value)

    # test training flag being set properly
    detect_training_io = {'inputs': ['input'], 'outputs': ['output']}
    detect_training_model = DetectTrainingModel(dataset=None, log_dir='', **detect_training_io)
    detect_training_batch = {'input': [[1]*10]}
    outputs = detect_training_model.run(detect_training_batch, train=False)
    assert np.allclose(outputs['output'], [[0]*10])
    outputs2 = detect_training_model.run(detect_training_batch, train=True)
    assert np.allclose(outputs2['output'], [[2]*10])


def test_run_bad_outputs():
    """Test if Exceptions are raised when bad output is encountered."""
    batch = {'input': [[1]*10], 'target': [[0]*10]}
    scalar_output_model = TrainableModel(dataset=None, log_dir='', inputs=['input', 'target'],
                                         outputs=['loss', 'scalar_output'], optimizer=_OPTIMIZER)
    with pytest.raises(ValueError):
        scalar_output_model.run(batch)  # scalar (non-batched) output


def test_run_custom_loss():
    CUSTOM_LOSS = 'custom_loss'
    IO_CUSTOM = {'inputs': ['input', 'target'], 'outputs': ['output', CUSTOM_LOSS]}

    # test variables update if and only if ``train=True``
    trainable_model = TrainableModel(dataset=None, log_dir='', **IO_CUSTOM, optimizer=_OPTIMIZER,
                                     loss_name=CUSTOM_LOSS)
    batch = {'input': [[1] * 10], 'target': [[0] * 10]}

    # single run with ``train=False``
    orig_value = trainable_model.var.eval(session=trainable_model.session)
    trainable_model.run(batch, train=False)
    after_value = trainable_model.var.eval(session=trainable_model.session)
    assert np.allclose(orig_value, after_value)

    # multiple runs with ``train=False``
    for _ in range(100):
        trainable_model.run(batch, train=False)
    after_value = trainable_model.var.eval(session=trainable_model.session)
    assert np.allclose(orig_value, after_value)

    # single run with ``train=True``
    trainable_model.run(batch, train=True)
    after_value = trainable_model.var.eval(session=trainable_model.session)
    assert not np.allclose(orig_value, after_value)

    # multiple runs with ``train=True``
    trainable_model.run(batch, train=True)
    for _ in range(1000):
        trainable_model.run(batch, train=True)
    after_value = trainable_model.var.eval(session=trainable_model.session)
    assert np.allclose([0] * 10, after_value)


def test_run_gradient_clipping():
    """Test gradient is clipped."""
    trainable_model = TrainableModel(dataset=None, log_dir='', **_IO, optimizer=_OPTIMIZER_CLIPPING, clip_gradient=1e-6)
    batch = {'input': [[1]*10], 'target': [[0]*10]}
    orig_value = trainable_model.var.eval(session=trainable_model.session)
    trainable_model.run(batch, train=True)
    after_value = trainable_model.var.eval(session=trainable_model.session)
    assert np.allclose(orig_value, after_value, atol=1.e-6)


def test_mainloop_model_training(tmpdir):
    """Test the model is being trained properly."""
    _, model, mainloop = create_simple_main_loop(130, tmpdir)
    mainloop.run_training()
    after_value = model.graph.get_tensor_by_name('var:0').eval(session=model.session)
    assert np.allclose([0]*10, after_value, atol=0.01)


def test_mainloop_zero_epoch_not_training(tmpdir):
    """Test the model is not being trained in the zeroth epoch."""
    _, model, mainloop = create_simple_main_loop(0, tmpdir)
    mainloop.run_training()
    after_value = model.graph.get_tensor_by_name('var:0').eval(session=model.session)
    assert np.allclose([2]*10, after_value, atol=0.01)


def test_restore_1(tmpdir):
    """Test restore from directory with one valid checkpoint."""
    # test model saving
    trainable_model = TrainableModel(dataset=None, log_dir=tmpdir, **_IO, optimizer=_OPTIMIZER_NO_MODULE)
    batch = {'input': [[1] * 10], 'target': [[0] * 10]}
    for _ in range(1000):
        trainable_model.run(batch, train=True)
    saved_var_value = trainable_model.var.eval(session=trainable_model.session)
    trainable_model.save('1')

    # test restoring
    restored_model = BaseModel(dataset=None, log_dir='', restore_from=tmpdir, **_IO, optimizer=_OPTIMIZER)

    var = restored_model.graph.get_tensor_by_name('var:0')
    var_value = var.eval(session=restored_model.session)
    assert np.allclose(saved_var_value, var_value)


def test_restore_2_with_spec(tmpdir):
    """Test restore from directory with two checkpoints where correct name is specified in the path"""
    # test model saving
    trainable_model = TrainableModel(dataset=None, log_dir=tmpdir, **_IO, optimizer=_OPTIMIZER)
    batch = {'input': [[1] * 10], 'target': [[0] * 10]}
    for _ in range(1000):
        trainable_model.run(batch, train=True)
    saved_var_value = trainable_model.var.eval(session=trainable_model.session)
    trainable_model.save('1')
    checkpoint_path = trainable_model.save('2')

    # test restoring
    restored_model = BaseModel(dataset=None, log_dir='', restore_from=checkpoint_path, **_IO, optimizer=_OPTIMIZER)

    var = restored_model.graph.get_tensor_by_name('var:0')
    var_value = var.eval(session=restored_model.session)
    assert np.allclose(saved_var_value, var_value)


def test_restore_2_without_spec(tmpdir):
    """Test restore from directory with two checkpoints and no specification of which one to restore from."""
    # test model saving
    trainable_model = TrainableModel(dataset=None, log_dir=tmpdir, **_IO, optimizer=_OPTIMIZER)
    batch = {'input': [[1] * 10], 'target': [[0] * 10]}
    for _ in range(1000):
        trainable_model.run(batch, train=True)
    trainable_model.save('1')
    trainable_model.save('2')

    # test restoring
    with pytest.raises(ValueError):
        BaseModel(dataset=None, log_dir='', restore_from=tmpdir, **_IO)


def test_restore_0(tmpdir):
    """Test restore from directory with no checkpoints."""
    # test model saving
    trainable_model = TrainableModel(dataset=None, log_dir=tmpdir, **_IO, optimizer=_OPTIMIZER)
    batch = {'input': [[1] * 10], 'target': [[0] * 10]}
    for _ in range(1000):
        trainable_model.run(batch, train=True)

    # test restoring
    with pytest.raises(ValueError):
        BaseModel(dataset=None, log_dir='', restore_from=tmpdir, **_IO)


def test_restore_and_train(tmpdir):
    """Test model training after restoring."""
    # save a model that is not trained
    trainable_model = TrainableModel(dataset=None, log_dir=tmpdir, **_IO, optimizer=_OPTIMIZER)
    trainable_model.save('')

    # restored the model
    restored_model = BaseModel(dataset=None, log_dir='', restore_from=tmpdir, **_IO)

    # test whether it can be trained
    batch = {'input': [[1] * 10], 'target': [[0] * 10]}
    for _ in range(1000):
        restored_model.run(batch, train=True)

    after_value = restored_model.graph.get_tensor_by_name('var:0').eval(session=restored_model.session)
    assert np.allclose([0]*10, after_value)


def test_model_monitoring(tmpdir):
    """Test the model monitoring works properly."""
    dataset = SimpleDataset()
    model = TrainableModel(dataset=dataset, log_dir=tmpdir, **_IO, optimizer=_OPTIMIZER)
    batch = next(iter(dataset.train_stream()))
    outputs = model.run(batch, False, None)
    assert BaseModel.SIGNAL_VAR_NAME not in outputs
    assert BaseModel.SIGNAL_MEAN_NAME not in outputs

    with pytest.raises(ValueError):
        TrainableModel(dataset=dataset, log_dir=tmpdir, **_IO, optimizer=_OPTIMIZER,
                       monitor='can_not_be_found')

    monitored_model = TrainableModel(dataset=dataset, log_dir=tmpdir, **_IO, optimizer=_OPTIMIZER,
                                     monitor='output')
    monitored_output = monitored_model.run(batch, False, None)
    assert np.allclose([2.]*_DATASET_SHAPE[0], monitored_output[BaseModel.SIGNAL_MEAN_NAME], atol=0.01)
    assert np.allclose([0.]*_DATASET_SHAPE[0], monitored_output[BaseModel.SIGNAL_VAR_NAME], atol=0.01)

    with pytest.raises(ValueError):
        TrainableModel(dataset=dataset, log_dir=tmpdir, inputs=['input', 'target', BaseModel.SIGNAL_MEAN_NAME],
                       outputs=['output', 'loss'], optimizer=_OPTIMIZER, monitor='output')
    with pytest.raises(ValueError):
        TrainableModel(dataset=dataset, log_dir=tmpdir, inputs=['input', 'target'], optimizer=_OPTIMIZER,
                       outputs=['output', 'loss', BaseModel.SIGNAL_VAR_NAME], monitor='output')


def test_regularization():
    """Test if tensors in REGULARIZATION_LOSSES collections are properly utilized for training."""
    regularized_model = RegularizedModel(dataset=None, log_dir='', **_IO, optimizer=_OPTIMIZER)
    batch = {'input': [[1]*10], 'target': [[0]*10]}

    with pytest.raises(tf.errors.InvalidArgumentError):  # placeholder ratio is required for computing the loss
        regularized_model.run(batch, train=True)

    regularized_model2 = RegularizedModel(dataset=None, log_dir='', inputs=['input', 'target', 'ratio'],
                                          outputs=['loss', 'output'], optimizer=_OPTIMIZER)
    good_batch = {'input': [[1]*10], 'target': [[0]*10], 'ratio': [1.0]}
    regularized_model2.run(good_batch, train=True)


def test_profiling(tmpdir):
    """Test whether profile is created."""
    model = TrainableModel(dataset=None, log_dir=tmpdir, **_IO, optimizer=_OPTIMIZER, profile=True, keep_profiles=10)
    batch = {'input': [[1]*10], 'target': [[0]*10]}

    # test if one can train one model while the other remains intact
    for _ in range(1000):
        model.run(batch, train=True)

    for i in range(10):
        assert path.exists(f"{tmpdir}/profile_{i}.json")

    assert not path.exists(f"{tmpdir}/profile_11.json")


#######################
# TF Base Model Saver #
#######################
"""Test case for correct usage of TF ``Saver`` in ``BaseModel``."""


def test_keep_checkpoints(tmpdir):
    """
    Test if the checkpoints are kept.

    This is regression test for issue #71 (TF ``Saver`` is keeping only the last 5 checkpoints).
    """
    dummy_model = SimpleModel(dataset=None, log_dir=tmpdir, inputs=[], outputs=['output'])

    checkpoints = []
    for i in range(20):
        checkpoints.append(dummy_model.save(str(i)))

    for checkpoint in checkpoints:
        assert path.exists(checkpoint+'.index')
        assert path.exists(checkpoint+'.meta')
        data_prefix = path.basename(checkpoint)+'.data'
        data_files = [file for file in os.listdir(path.dirname(checkpoint)) if file.startswith(data_prefix)]
        assert len(data_files) > 0


def test_freeze(tmpdir):
    """
    Test if the checkpoints are kept.

    This is regression test for issue #71 (TF ``Saver`` is keeping only the last 5 checkpoints).
    """
    dummy_model = SimpleModel(dataset=None, log_dir=tmpdir, inputs=[], outputs=['output'], freeze=True)
    checkpoint = dummy_model.save('')

    assert path.exists(checkpoint+'.index')
    assert path.exists(checkpoint+'.meta')
    assert path.exists(checkpoint[:-4]+'pb')

    with pytest.raises(ValueError):
        freeze_graph(input_graph='does_not_exists.graph', input_checkpoint=checkpoint, output_node_names=[],
                     output_graph=path.join(tmpdir, 'out.pb'))
    with pytest.raises(ValueError):
        freeze_graph(input_graph=checkpoint[:-4]+'graph', input_checkpoint='does_not_exists.ckpt',
                     output_node_names=[], output_graph=path.join(tmpdir, 'out.pb'))


######################
# TF Base Model MGMT #
######################
"""Test case for correct management of TF graphs and sessions."""


def test_two_models_created():
    """
    Test if one can create and train two ``BaseModels``.

    This is regression test for issue #83 (One can not create and use more than one instance of ``BaseModel``).
    """
    model1 = TrainableModel(dataset=None, log_dir='', **_IO, optimizer=_OPTIMIZER)
    model2 = TrainableModel(dataset=None, log_dir='', **_IO, optimizer=_OPTIMIZER)
    batch = {'input': [[1]*10], 'target': [[0]*10]}

    # test if one can train one model while the other remains intact
    for _ in range(1000):
        model1.run(batch, train=True)
    trained_value = model1.var.eval(session=model1.session)
    assert np.allclose([0]*10, trained_value)
    default_value = model2.var.eval(session=model2.session)
    assert np.allclose([2]*10, default_value)

    # test if one can train the other model
    for _ in range(1000):
        model2.run(batch, train=True)
    trained_value2 = model2.var.eval(session=model2.session)
    assert np.allclose([0] * 10, trained_value2)


def test_two_models_restored(tmpdir):
    """
    Test if one can ``_restore_model`` and use two ``BaseModels``.

    This is regression test for issue #83 (One can not create and use more than one instance of ``BaseModel``).
    """
    tmpdir2 = tempfile.mkdtemp()

    model1 = TrainableModel(dataset=None, log_dir=tmpdir, **_IO, optimizer=_OPTIMIZER)
    model2 = TrainableModel(dataset=None, log_dir=tmpdir2, **_IO, optimizer=_OPTIMIZER)
    batch = {'input': [[1] * 10], 'target': [[0] * 10]}
    for _ in range(1000):
        model1.run(batch, train=True)

    model1.save('')
    model2.save('')

    # test if one can ``_restore_model`` two models and use them at the same time
    restored_model1 = BaseModel(dataset=None, log_dir='', restore_from=tmpdir, **_IO)
    restored_model2 = BaseModel(dataset=None, log_dir='', restore_from=tmpdir2, **_IO)

    trained_value = restored_model1.graph.get_tensor_by_name('var:0').eval(session=restored_model1.session)
    assert np.allclose([0]*10, trained_value)
    default_value = restored_model2.graph.get_tensor_by_name('var:0').eval(session=restored_model2.session)
    assert np.allclose([2]*10, default_value)

    shutil.rmtree(tmpdir2)


###########################
# TF Base Model Multi GPU #
###########################
"""Test case for correct handling of multi-gpu trainings."""


def test_incomplete_batches():
    """Test if incomplete batches are handled properly in multi-tower env."""
    multi_gpu_model = TrainableModel(dataset=None, log_dir='', **_IO, n_gpus=4,
                                     session_config={'allow_soft_placement': True}, optimizer=_OPTIMIZER)
    batch = {'input': [[1]*10]*8, 'target': [[0]*10]*8}
    small_batch = {'input': [[1]*10]*3, 'target': [[0]*10]*3}

    # single run with full batch
    outputs = multi_gpu_model.run(batch, train=False)
    assert np.allclose(outputs['output'], [[2]*10]*8)

    # single run with small batch
    outputs2 = multi_gpu_model.run(small_batch, train=False)
    assert np.allclose(outputs2['output'], [[2]*10]*3)

    # multiple train runs with full batch
    for _ in range(1000):
        multi_gpu_model.run(batch, train=True)
    after_value = multi_gpu_model.var.eval(session=multi_gpu_model.session)
    assert np.allclose(after_value, [0]*10)

    multi_gpu_model2 = TrainableModel(dataset=None, log_dir='', **_IO, n_gpus=4,
                                      session_config={'allow_soft_placement': True}, optimizer=_OPTIMIZER)
    # multiple train runs with small batch
    for _ in range(1000):
        multi_gpu_model2.run(small_batch, train=True)
    after_value = multi_gpu_model2.var.eval(session=multi_gpu_model2.session)
    assert np.allclose(after_value, [0]*10)
