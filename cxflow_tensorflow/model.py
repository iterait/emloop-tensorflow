"""
Module with cxflow trainable models defined in TensorFlow, which are restorable from their corresponding checkpoints.

Provides `BaseModel` which manages model config, API and unifies the TF graph <=> cxflow touch points.
"""
import math
import logging
from os import path
from abc import ABCMeta
from typing import List, Mapping, Optional
from glob import glob

import tensorflow as tf
import numpy as np

from cxflow import AbstractModel, AbstractDataset

from .third_party.tensorflow.freeze_graph import freeze_graph
from .third_party.tensorflow.average_gradients import average_gradients
from .utils import create_optimizer


class GraphTower:
    """
    GraphTower is a lightweight wrapper around a tower (TF sub-graph) in multi-GPU models.
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

    def __init__(self, id_: int, inputs: List[str], outputs: List[str]):
        self._id = id_
        self._device_name = '/cpu:0' if id_ < 0 else '/gpu:{}'.format(id_)
        self._input_names = inputs
        self._output_names = outputs
        self._inputs = {}
        self._outputs = {}
        self._loss = None

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
    def loss(self):
        return self['loss']

    @property
    def inputs(self):
        return self._inputs.values()

    @property
    def outputs(self):
        return self._outputs.values()

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_names(self):
        return self._output_names

    @property
    def batch_size(self):
        return tf.shape(self[self._input_names[0]])[0]

    def __getitem__(self, item):
        if item in self._outputs:
            return self._outputs[item]
        elif item in self._inputs:
            return self._inputs[item]
        else:
            raise KeyError('Tensor `{}` is not within the input/output tensors'.format(item))

    def __enter__(self):
        self._device = tf.device(self._device_name)
        self._device.__enter__()

    def __exit__(self, *args):
        self._device.__exit__(*args)


class BaseModel(AbstractModel, metaclass=ABCMeta):   # pylint: disable=too-many-instance-attributes
    """
    Base TensorFlow model enforcing uniform model API which is trainable in cxflow main loop.

    All tf models should be derived from this class and override `_create_model` method.
    Optionally, `_restore_model` might be overriden.
    """

    def __init__(self,  # pylint: disable=too-many-arguments
                 dataset: Optional[AbstractDataset], log_dir: str, inputs: List[str], outputs: List[str],
                 n_gpus: int=0, restore_from: Optional[str]=None, restore_model_name: Optional[str]=None,
                 optimizer=None, freeze=False, **kwargs):
        """
        Create a cxflow trainable TensorFlow model.

        In case `restore_from` is not `None`, the model will be restored from a checkpoint. See `_restore_model`
        for more information.

        :param dataset: dataset to be trained with
        :param log_dir: path to the logging directory (wherein models should be saved)
        :param inputs: model input names
        :param outputs: model output names
        :param device: tf device to be trained on
        :param threads: number of threads to be used by tf
        :param restore_from: path to directory from which the model is restored
        :param restore_model_name: model name to be restored (e.g. `model.ckpt`)
        :param kwargs: additional kwargs which are passed to the _create_model method
        """
        super().__init__(dataset=dataset, log_dir=log_dir, restore_from=restore_from)

        self._dataset = dataset
        self._log_dir = log_dir
        self._freeze_graph = freeze
        self._train_op = None
        self._graph = self._saver = None
        self._towers = [GraphTower(i, inputs, outputs) for i in range(n_gpus)]
        if n_gpus == 0:
            self._towers.append(GraphTower(-1, inputs, outputs))

        logging.info('\tCreating TF model on %s devices', n_gpus)
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)
        with self._graph.as_default():
            if restore_from is not None:
                self._restore_model(restore_from=restore_from, restore_model_name=restore_model_name)
            else:
                with tf.variable_scope(tf.get_variable_scope()) as scope:
                    for tower in self._towers:
                        with tower:
                            self._create_model(**kwargs)
                        scope.reuse_variables()

            for tower in self._towers:
                tower.find_io_tensors()

            if restore_from is None:
                logging.debug('\tCreating train op')
                self._create_train_op(optimizer)

                logging.debug('\tInitializing the variables')
                self._session.run(tf.global_variables_initializer())
                self._session.run(tf.local_variables_initializer())

            logging.debug('\tFinding train_op in the created graph')
            try:
                self._train_op = self._graph.get_operation_by_name('train_op')
            except (KeyError, ValueError, TypeError) as ex:
                raise ValueError('Cannot find train op in graph. Train op must be named `train_op`.') from ex

            logging.debug('\tCreating Saver')
            self._saver = tf.train.Saver(max_to_keep=100000000)

    @property
    def input_names(self) -> List[str]:   # pylint: disable=invalid-sequence-index
        """List of TF tensor names listed as model inputs."""
        return self._towers[0].input_names

    @property
    def output_names(self) -> List[str]:   # pylint: disable=invalid-sequence-index
        """List of TF tensor names listed as model outputs."""
        return self._towers[0].output_names

    @property
    def graph(self) -> tf.Graph:
        """TF graph object."""
        return self._graph

    @property
    def session(self) -> tf.Session:
        """TF session object."""
        return self._session

    def run(self, batch: Mapping[str, object], train: bool) -> Mapping[str, object]:
        """
        Run the model with the given batch as feed_dict. Update the trainable variables only if train is true.
        Fetch and return all the model outputs as a dict.
        :param batch: batch dict source_name->values
        :param train: flag whether parameters update (train_op) should be included in fetches
        :return: outputs dict
        """
        # setup the feed dict
        batch_size = len(batch[next(iter(batch))])
        tower_batch_size = math.ceil(batch_size / len(self._towers))

        feed_dict = {}
        fetches = [self._train_op] if train else []

        for i, tower in enumerate(self._towers):
            for placeholder_name in self.input_names:
                tower_batch = batch[placeholder_name][i*tower_batch_size:(i+1)*tower_batch_size]
                feed_dict[tower[placeholder_name]] = tower_batch
            for output_name in self.output_names:
                fetches.append(tower[output_name])

        # run the computational graph for one batch
        outputs = self._session.run(fetches=fetches, feed_dict=feed_dict)

        if train:
            outputs = outputs[1:]

        # stack partial tower outputs
        num_outputs = len(self.output_names)
        stacked_outputs = [np.concatenate(outputs[i::num_outputs]) for i in range(num_outputs)]

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

    def _restore_checkpoint(self, checkpoint_name: str) -> None:
        """
        Given the checkpoint name (including the path to it), restore the model.

        :param checkpoint_name: name in form of `*.ckpt`, e.g. `model_3.ckpt`.
        """
        logging.debug('Loading meta graph')
        saver = tf.train.import_meta_graph(checkpoint_name + '.meta')
        logging.debug('Restoring model')
        saver.restore(self._session, checkpoint_name)

    def _restore_model(self, restore_from: str, restore_model_name: Optional[str]=None) -> None:
        """
        Restore TF model from the given checkpoint.
        :param restore_from: path to directory from which the model is restored
        :param restore_model_name: model name to be restored (e.g. `model.ckpt`)
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
    def restore_fallback_module(self) -> str:
        return 'cxflow_tensorflow'

    @property
    def restore_fallback_class(self) -> str:
        return 'BaseModel'

    def _create_train_op(self, optimizer_config: Optional[dict]) -> None:
        """
        TODO: docstring
        :param kwargs: model configuration
        """
        grads_and_vars = []
        optimizer = create_optimizer(optimizer_config)
        for tower in self._towers:
            with tower:
                grads_and_vars.append(optimizer.compute_gradients(tower.loss))

        optimizer.apply_gradients(average_gradients(grads_and_vars), name='train_op')

    def _create_model(self, **kwargs) -> None:
        """
        TODO: docstring
        :param kwargs: model configuration
        """
        raise NotImplementedError('`_create_model` method must be implemented in order to construct a new model.')
