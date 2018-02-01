import math
import logging
from os import path
from abc import ABCMeta
from typing import List, Mapping, Optional, Iterable
from glob import glob

import numpy as np
import cxflow as cx
import tensorflow as tf

from .third_party.tensorflow.freeze_graph import freeze_graph
from .third_party.tensorflow.average_gradients import average_gradients
from .utils import create_optimizer


DEFAULT_LOSS_NAME = 'loss'
"""Default loss tensor name."""


class GraphTower:
    """
    ``GraphTower`` is a lightweight wrapper around a tower (TF sub-graph) in multi-GPU models.
    It allows to work with multiple copies of the same sub-graph distributed on multiple devices
    with only one set of input and output names.

    ---------------------------------------------
    USAGE
    ---------------------------------------------
    1. create the desired number of GraphTowers:
        towers = [GraphTower(i, inputs, outputs) for i in range(4)]
    2. create the TF sub-graphs in the tower environments (uses with tf.device(...)):
        for tower in towers:
            with tower:
                # define the TF graph with the respective inputs and outputs
    3. find the input placeholders and output variables:
        for tower in towers:
            tower.find_io_tensors()
    4. access the io tensors, loss etc.
        towers[3]['my_input']  # my_input placeholder which is actually named 'my_input_3:0'

    ---------------------------------------------
    WARNING
    ---------------------------------------------
    The sub-graphs must be defined in the order corresponding to the tower ids!
    """

    def __init__(self, id_: int, inputs: List[str], outputs: List[str], loss_name: str=DEFAULT_LOSS_NAME):
        """
        Create new GraphTower.
        :param id_: tower (gpu) id, towers with negative ids are placed on /cpu:0
        :param inputs: tower input names
        :param outputs: tower output names
        :param loss_name: loss tensor name
        """
        self._id = id_
        self._device_name = '/cpu:0' if id_ < 0 else '/gpu:{}'.format(id_)
        self._input_names = inputs
        self._output_names = outputs
        self._inputs = {}
        self._outputs = {}
        self._loss_name = loss_name

    def _get_full_name(self, tensor_name: str) -> str:
        """
        Translates a simple tensor name to the actual tensor name in the sub-graph.

        E.g.:
        variable named `loss` in the 0th tower will be named `loss:0`
        variable name `predictions` in the 1st tower will be name `predictions_1:0`
        """
        return tensor_name + ('' if self._id < 1 else '_{}'.format(self._id)) + ':0'

    def _find_or_raise(self, tensor_name: str) -> tf.Tensor:
        """
        Find the tensor with the given name in the default graph or raise an exception.
        :param tensor_name: tensor name to be find
        :return: tf.Tensor
        """
        full_name = self._get_full_name(tensor_name)
        try:
            return tf.get_default_graph().get_tensor_by_name(full_name)
        except (KeyError, ValueError, TypeError) as ex:
            raise ValueError('Tensor `{}` with full name `{}` was not found.'.format(tensor_name, full_name)) from ex

    def find_io_tensors(self) -> None:
        """Find the tower's input and output tensors in the default graph."""
        for input_name in self._input_names:
            self._inputs[input_name] = self._find_or_raise(input_name)
        for output_name in self._output_names:
            self._outputs[output_name] = self._find_or_raise(output_name)

    @property
    def loss(self) -> tf.Tensor:
        """Return the loss tensor."""
        return self[self._loss_name]

    @property
    def inputs(self) -> Iterable[tf.Tensor]:
        """Return an iterable collection of input tensors."""
        return self._inputs.values()

    @property
    def outputs(self) -> Iterable[tf.Tensor]:
        """Return an iterable collection of output tensors."""
        return self._outputs.values()

    @property
    def input_names(self) -> List[str]:
        """Return list of the input names."""
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        """Return list of the output names."""
        return self._output_names

    @property
    def batch_size(self) -> tf.Tensor:
        """Return the current batch size."""
        return tf.shape(self[self._input_names[0]])[0]

    def __getitem__(self, item) -> tf.Tensor:
        """Return input/output tensor with the given name."""
        if item in self._outputs:
            return self._outputs[item]
        elif item in self._inputs:
            return self._inputs[item]
        else:
            raise KeyError('Tensor `{}` is not within the input/output tensors'.format(item))

    def __enter__(self) -> None:
        """Enter with tf.device(...): env."""
        self._device = tf.device(self._device_name)
        self._device.__enter__()

    def __exit__(self, *args) -> None:
        """Exit with tf.device(...): env."""
        self._device.__exit__(*args)


class BaseModel(cx.AbstractModel, metaclass=ABCMeta):  # pylint: disable=too-many-instance-attributes
    """
    Cxflow :py:class:`AbstractModel <cxflow.models.AbstractModel>` implementation for TensorFlow models.

    To define a **cxflow** trainable model in TensorFlow, derive your class from :py:class:`BaseModel` and override
    :py:meth:`_create_model` method.

    See the method references for additional customization options.
    """

    TRAIN_OP_NAME = 'train_op'
    """Expected train op tensor name prefix."""

    TRAINING_FLAG_NAME = 'cxf_is_training'
    """Training flag variable name."""

    def __init__(self,  # pylint: disable=too-many-arguments
                 dataset: Optional[cx.AbstractDataset], log_dir: str, inputs: List[str], outputs: List[str],
                 session_config: Optional[dict]=None, n_gpus: int=0, restore_from: Optional[str]=None,
                 restore_model_name: Optional[str]=None, optimizer=None, freeze=False, loss_name: str=DEFAULT_LOSS_NAME,
                 **kwargs):
        """
        Create new cxflow trainable TensorFlow model.

        TF Graph, train ops etc. are constructed with the following procedure:

        #. Create ``tf.Graph`` and ``tf.Session`` with :py:meth:`_create_session`
        #. Either create or restore the model with :py:meth:`_create_model` or :py:meth:`_restore_model` respectively
        #. Find input/output tensors
        #. Create train ops with :py:meth:`_create_train_ops` unless they are already restored
        #. Find the train ops
        #. Create ``tf.Saver``

        .. note::
            In most cases, it is not required to re-define the ``__init__`` method for your models.

        :param dataset: dataset to be trained with
        :param log_dir: path to the logging directory (wherein models should be saved)
        :param inputs: model input names
        :param outputs: model output names
        :param session: TF session configuration dict, see :py:meth:`_create_session`
        :param n_gpus: number of GPUs to use
        :param restore_from: path to directory from which the model is restored
        :param restore_model_name: model name to be restored (e.g. ``model.ckpt``)
        :param optimizer: TF optimizer configuration dict
        :param freeze: freeze the graph after each save
        :param loss_name: loss tensor name
        :param kwargs: additional kwargs forwarded to :py:meth:`_create_model`
        """
        super().__init__(dataset=dataset, log_dir=log_dir, restore_from=restore_from)

        self._dataset = dataset
        self._log_dir = log_dir
        self._freeze_graph = freeze
        self._loss_name = loss_name
        self._train_ops = []
        self._graph = self._saver = None
        self._towers = [GraphTower(i, inputs, outputs, loss_name) for i in range(n_gpus)]
        if n_gpus == 0:
            self._towers.append(GraphTower(-1, inputs, outputs, loss_name))

        logging.info('\tCreating TF model on %s GPU devices', n_gpus)
        self._graph = tf.Graph()
        self._session = self._create_session(session_config)
        dependencies = []
        with self._graph.as_default():
            if restore_from is None:
                with tf.variable_scope(tf.get_variable_scope()) as scope:
                    self._is_training = tf.placeholder(tf.bool, [], BaseModel.TRAINING_FLAG_NAME)
                    for tower in self._towers:
                        with tower:
                            self._create_model(**kwargs)
                        dependencies.append(list(self._graph.get_collection(tf.GraphKeys.UPDATE_OPS)))
                        scope.reuse_variables()
            else:
                self._restore_model(restore_from=restore_from, restore_model_name=restore_model_name)
                try:
                    self._is_training = self._graph.get_tensor_by_name(BaseModel.TRAINING_FLAG_NAME + ':0')
                except (KeyError, ValueError, TypeError):
                    logging.warning('Could not find training flag placeholder in the graph, creating a new one.')
                    self._is_training = tf.placeholder(tf.bool, [], BaseModel.TRAINING_FLAG_NAME)

            for tower in self._towers:
                tower.find_io_tensors()

            if restore_from is None:
                logging.debug('\tCreating train ops')
                self._create_train_ops(dependencies, optimizer)

            logging.debug('\tCreating Saver')
            self._saver = tf.train.Saver(max_to_keep=100000000)

            if restore_from is None:
                logging.debug('\tInitializing the variables')
                self._initialize_variables(**kwargs)

            logging.debug('\tSearching for the train ops in the created graph')
            try:
                for i in range(len(self._towers)):
                    self._train_ops.append(
                        self._graph.get_operation_by_name(BaseModel.TRAIN_OP_NAME+'_{}'.format(i + 1)))
            except (KeyError, ValueError, TypeError) as ex:
                raise ValueError('Cannot find train op {} in graph. '
                                 'The op must be named `train_op_{}`.'.format(i+1, i+1)) from ex

            train_vars = self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            logging.debug('Trainable variables: %s', [var.name for var in train_vars])
            logging.info('Number of parameters: %s', sum([np.prod(var.get_shape().as_list()) for var in train_vars]))

    @property
    def input_names(self) -> List[str]:  # pylint: disable=invalid-sequence-index
        """List of TF input tensor (placeholder) names."""
        return self._towers[0].input_names

    @property
    def output_names(self) -> List[str]:  # pylint: disable=invalid-sequence-index
        """List of TF output tensor names."""
        return self._towers[0].output_names

    @property
    def is_training(self) -> tf.Tensor:
        """
        Training flag tensor.

        This is useful for determining whether to use certain ops such as dropout.
        """
        return self._is_training

    @property
    def graph(self) -> tf.Graph:
        """TF graph object."""
        return self._graph

    @property
    def session(self) -> tf.Session:
        """TF session object."""
        return self._session

    def run(self, batch: cx.Batch, train: bool, stream: cx.datasets.StreamWrapper=None) -> Mapping[str, object]:
        """
        Run the model with the given ``batch``. Update the trainable variables only if ``train`` is true.

        Fetch and return all the model outputs as a dict.

        :param batch: batch dict ``{source_name: values}``
        :param train: flag whether parameters update (``train_op``) should be included in fetches
        :param stream: stream wrapper (useful for precise buffer management)
        :raise ValueError: if an output is wrongly typed or its batch size differs from the input batch size
        :return: outputs dict
        """
        # setup the feed dict
        batch_size = len(batch[next(iter(batch))])
        tower_batch_size = math.ceil(batch_size / len(self._towers))
        nonempty_towers = batch_size // tower_batch_size + (0 if batch_size % tower_batch_size == 0 else 1)

        feed_dict = {self._is_training: train}
        fetches = [self._train_ops[nonempty_towers-1]] if train else []

        for i, tower in enumerate(self._towers):
            if i*tower_batch_size < batch_size:
                for placeholder_name in self.input_names:
                    tower_batch = batch[placeholder_name][i*tower_batch_size:(i+1)*tower_batch_size]
                    feed_dict[tower[placeholder_name]] = tower_batch
                for output_name in self.output_names:
                    fetches.append(tower[output_name])

        # run the computational graph for one batch and allow buffering in the meanwhile
        if stream is not None:
            with stream.allow_buffering:
                outputs = self._session.run(fetches=fetches, feed_dict=feed_dict)
        else:
            outputs = self._session.run(fetches=fetches, feed_dict=feed_dict)

        if train:
            outputs = outputs[1:]

        for i, output in enumerate(outputs):
            if not isinstance(output, (list, tuple, np.ndarray)):
                output_name = self.output_names[i % len(self.output_names)]
                raise ValueError('Model output `{}` is not one of list, tuple or numpy array. Found `{}` instead. '
                                 'Model outputs should be batched, i.e. the first dimension should refer to '
                                 'different examples.'.format(output_name, type(output)))

        # stack partial tower outputs
        num_outputs = len(self.output_names)
        stacked_outputs = [np.concatenate(outputs[i::num_outputs]) for i in range(num_outputs)]

        for i, output in enumerate(stacked_outputs):
            if len(output) != batch_size:
                raise ValueError('Input-output batch size mismatch. Input: {} Output: {} for `{}`'
                                 .format(batch_size, len(output), self.output_names[i]))

        return dict(zip(self.output_names, stacked_outputs))

    def save(self, name_suffix: str) -> str:
        """
        Save current tensorflow graph to a checkpoint named with the given name suffix.

        The checkpoint will be locaced in self.log_dir directory.
        :param name_suffix: saved checkpoint name suffix
        :return: path to the saved checkpoint
        """
        graph_path = path.join(self._log_dir, 'model_{}.graph'.format(name_suffix))
        checkpoint_path = path.join(self._log_dir, 'model_{}.ckpt'.format(name_suffix))
        frozen_graph_path = path.join(self._log_dir, 'model_{}.pb'.format(name_suffix))

        tf.train.write_graph(self._session.graph_def, '', graph_path, as_text=False)

        self._saver.save(self._session, checkpoint_path)

        if self._freeze_graph:
            with tf.Graph().as_default():
                freeze_graph(input_graph=graph_path,
                             input_checkpoint=checkpoint_path,
                             output_node_names=self.output_names,
                             output_graph=frozen_graph_path)

        return checkpoint_path

    def _restore_checkpoint(self, checkpoint_path: str) -> None:
        """
        Restore model from the given ``checkpoint_path``.

        :param checkpoint_path: full path to the checkpoint, e.g. ``my_dir/model_3.ckpt``.
        """
        logging.debug('Loading meta graph')
        saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
        logging.debug('Restoring model')
        saver.restore(self._session, checkpoint_path)

    def _restore_model(self, restore_from: str, restore_model_name: Optional[str]=None) -> None:
        """
        Restore TF model from the given ``restore_from`` path and ``restore_model_name``.

        The model name can be derived if the ``restore_from`` directory contains exactly one checkpoint.

        :param restore_from: path to directory from which the model is restored
        :param restore_model_name: model name to be restored (e.g. ``model.ckpt``)
        """

        logging.info('Restoring model from `{}`'.format(restore_from))
        assert path.isdir(restore_from), '`BaseModel` expect `restore_from` to be an existing directory.'
        meta_files = glob('{}/*.ckpt.meta'.format(restore_from))

        if len(meta_files) == 0:
            raise ValueError('No `{}/*.ckpt.meta` files found.'.format(restore_from))
        elif len(meta_files) == 1:
            logging.info('Restoring model from checkpoint metafile`{}`'.format(meta_files[0]))
            self._restore_checkpoint(meta_files[0][:-5])
        else:
            logging.info('Multiple checkpoint metafiles found.')

            if restore_model_name is None:
                raise ValueError('There are multiple checkpoint metafiles found in the directory {}. However, config '
                                 'lacks `model.restore_model_name`. Please, specify it.'.format(restore_from))

            logging.info('Restoring model from checkpoint `{}` located in directory `{}`'.format(restore_model_name,
                                                                                                 restore_from))
            self._restore_checkpoint(path.join(restore_from, restore_model_name))

    @property
    def restore_fallback(self) -> str:
        return 'cxflow_tensorflow.BaseModel'

    def _create_session(self, session_config: Optional[dict]) -> tf.Session:
        """
        Create and return TF Session for this model.

        By default the session is configured with ``tf.ConfigProto`` created with
        the given ``session_config`` as ``**kwargs``. Nested dictionaries such as
        ``gpu_options`` or ``graph_options`` are handled automatically.

        :param session_config: session configuration dict as specified in the config yaml
        :return: TensorFlow session
        """
        if session_config:
            session_config = tf.ConfigProto(**session_config)
        return tf.Session(graph=self._graph, config=session_config)

    def _create_train_ops(self, dependencies: List[List[tf.Operation]], optimizer_config: Optional[dict]) -> None:
        """
        Create the train ops for training. In order to handle incomplete batches, there must be one train op for
        each number of empty towers. E.g. for 2 GPU training, one must define 2 train ops for 1 and 2 towers
        respectively. The train ops must be named ``train_op_1``, ``train_op_2`` etc.
        wherein the suffixed number stands for the number of towers.

        By default the train ops are constructed in the following way:
            - optimizer is created from the ``model.optimizer`` configuration dict
            - REGULARIZATION_LOSSSES collection is summed to ``regularization_loss``
            - gradients minimizing the respective tower losses and ``regularization_loss`` are computed
            - for each number of non-empty towers
                - gradients of the respective towers are averaged and applied

        To implement a custom behavior, override this method and create your own op named as :py:attr:`TRAIN_OP_NAME`.

        .. code-block:: yaml
            :caption: example optimizer config

            model:
                optimizer:
                    class: RMSPropOptimizer
                    learning_rate: 0.001

        :param dependencies: a list of dependent operations (e.g. batch normalization updates) for each number of towers
        :param optimizer_config: optimizer configuration dict
        """
        if optimizer_config is None:
            raise ValueError('Optimizer config was not specified although it is required for creating the train op. '
                             'Please specify the configuration in `model.optimizer`.')
        grads_and_vars = []
        optimizer = create_optimizer(optimizer_config)
        regularization_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularization_loss = tf.reduce_sum(tf.stack(regularization_losses))
        if regularization_losses:
            logging.info('\tAdding regularization losses')
            logging.debug('\tRegularization losses: %s', [var.name for var in regularization_losses])
        for tower in self._towers:
            with tower:
                grads_and_vars.append(optimizer.compute_gradients(tf.reduce_mean(tower.loss) + regularization_loss))
        for i in range(len(self._towers)):
            with tf.control_dependencies(dependencies[i]):
                optimizer.apply_gradients(average_gradients(grads_and_vars[:(i + 1)]),
                                          name=BaseModel.TRAIN_OP_NAME + '_{}'.format(i + 1))

    def _create_model(self, **kwargs) -> None:
        """
        Create your TensorFlow model.

        Every model has to define:

        - loss tensor named according to given ``loss_name``
        - input placeholders and output tensors named according to the specified input and output names

        .. warning::
            To support multi-GPU training, all the variables must be created with ``tf.get_variable``
            and appropriate variable scopes.

        :param kwargs: model configuration as specified in ``model`` section of the configuration file
        """
        raise NotImplementedError('`_create_model` method must be implemented in order to construct a new model.')

    def _initialize_variables(self, **kwargs) -> None:
        """
        Initialize variables of your TensorFlow model.

        By default variables are initialized randomly.

        .. tip::

            Override this method to load variables from some check-point and fine-tune the model.

        :param kwargs: model configuration as specified in ``model`` section of the configuration file
        """
        self._session.run(tf.global_variables_initializer())
        self._session.run(tf.local_variables_initializer())
