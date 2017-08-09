"""
Module with cxflow trainable nets defined in tensorflow.

Provides BaseTFNet which manages net config, api and unifies tf graph <=> cxflow touch points.

Furthermore, this module exposes BaseTFNetRestore class
which is able to restore arbitrary cxflow nets from tf checkpoint.
"""
import logging
from os import path
from abc import ABCMeta
from typing import List, Mapping, Optional
from glob import glob

import tensorflow as tf

from cxflow import AbstractNet, AbstractDataset

from .third_party.tensorflow.freeze_graph import freeze_graph


class BaseTFNet(AbstractNet, metaclass=ABCMeta):   # pylint: disable=too-many-instance-attributes
    """
    Base TensorFlow network enforcing uniform net API which is trainable in cxflow main loop.

    All tf nets should be derived from this class and override `_create_net` method.
    Optionally, `_restore_net` might be overriden.
    """

    def __init__(self,  # pylint: disable=too-many-arguments
                 dataset: Optional[AbstractDataset], log_dir: str, inputs: List[str], outputs: List[str],
                 device: str='/cpu:0', threads: int=4, restore_from: Optional[str]=None,
                 restore_model_name: Optional[str]=None, **kwargs):
        """
        Create a cxflow trainable TensorFlow net.

        In case `restore_from` is not `None`, the network will be restored from a checkpoint. See `_restore_network`
        for more information.

        :param dataset: dataset to be trained with
        :param log_dir: path to the logging directory (wherein models should be saved)
        :param inputs: net input names
        :param outputs: net output names
        :param device: tf device to be trained on
        :param threads: number of threads to be used by tf
        :param restore_from: path to directory from which the model is restored
        :param restore_model_name: model name to be restored (e.g. `model.ckpt`)
        :param kwargs: additional kwargs which are passed to the _create_net method
        """
        super().__init__(dataset=dataset, log_dir=log_dir, restore_from=restore_from)

        assert outputs
        assert threads > 0

        self._dataset = dataset
        self._log_dir = log_dir
        self._train_op = None
        self._graph = self._saver = None
        self._input_names = inputs
        self._output_names = outputs
        self._tensors = {}

        with tf.device(device):
            logging.debug('Creating session')
            self._graph = tf.Graph()
            self._session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                             intra_op_parallelism_threads=threads,
                                                             allow_soft_placement=True),
                                       graph=self._graph)

            with self._graph.as_default():
                logging.debug('Creating net')
                if restore_from is not None:
                    self._restore_network(restore_from=restore_from, restore_model_name=restore_model_name)
                else:
                    self._create_net(**kwargs)

                logging.debug('Finding train_op in the created graph')
                try:
                    self._train_op = self._graph.get_operation_by_name('train_op')
                except (KeyError, ValueError, TypeError) as ex:
                    raise ValueError('Cannot find train op in graph. Train op must be named `train_op`.') from ex

                logging.debug('Finding io tensors in the created graph')
                for tensor_name in set(self._input_names + self._output_names):
                    full_name = tensor_name + ':0'
                    try:
                        tensor = self._graph.get_tensor_by_name(full_name)
                    except (KeyError, ValueError, TypeError) as ex:
                        raise ValueError('Tensor `{}` defined as input/output was not found. It has to be named `{}`.'
                                         .format(tensor_name, full_name)) from ex

                    if tensor_name not in self._tensors:
                        self._tensors[tensor_name] = tensor

                if not self._saver:
                    logging.debug('Creating Saver')
                    self._saver = tf.train.Saver(max_to_keep=100000000)

    @property
    def input_names(self) -> List[str]:   # pylint: disable=invalid-sequence-index
        """List of tf tensor names listed as net inputs."""
        return self._input_names

    @property
    def output_names(self) -> List[str]:   # pylint: disable=invalid-sequence-index
        """List of tf tensor names listed as net outputs."""
        return self._output_names

    @property
    def graph(self) -> tf.Graph:
        """Tf graph object."""
        return self._graph

    @property
    def session(self) -> tf.Session:
        """Tf session object."""
        return self._session

    @property
    def train_op(self) -> tf.Operation:
        """Net train op."""
        return self._train_op

    def get_tensor_by_name(self, name) -> tf.Tensor:
        """
        Get the tf tensor with the given name.

        Only tensor previously defined as net inputs/outputs in net.io can be accessed.
        :param name: tensor name
        :return: tf tensor
        """
        if name in self._tensors:
            return self._tensors[name]
        else:
            raise KeyError('Tensor named `{}` is not within accessible tensors.'.format(name))

    def run(self, batch: Mapping[str, object], train: bool) -> Mapping[str, object]:
        """
        Feed-forward the net with the given batch as feed_dict.
        Fetch and return all the net outputs as a dict.
        :param batch: batch dict source_name->values
        :param train: flag whether parameters update (train_op) should be included in fetches
        :return: outputs dict
        """
        # setup the feed dict
        feed_dict = {}
        for placeholder_name in self._input_names:
            feed_dict[self.get_tensor_by_name(placeholder_name)] = batch[placeholder_name]

        # setup fetches
        fetches = [self._train_op] if train else []
        for output_name in self._output_names:
            fetches.append(self.get_tensor_by_name(output_name))

        # run the computational graph for one batch
        batch_res = self._session.run(fetches=fetches, feed_dict=feed_dict)

        if train:
            batch_res = batch_res[1:]

        return dict(zip(self._output_names, batch_res))

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

        with tf.Graph().as_default():
            freeze_graph(input_graph=graph_path,
                         input_checkpoint=checkpoint_path,
                         output_node_names=self._output_names,
                         output_graph=frozen_graph_path)

        return checkpoint_path

    def _restore_checkpoint(self, checkpoint_name: str) -> None:
        """
        Given the checkpoint name (including the path to it), restore the network.

        :param checkpoint_name: name in form of `*.ckpt`, e.g. `model_3.ckpt`.
        """
        logging.debug('Loading meta graph')
        saver = tf.train.import_meta_graph(checkpoint_name + '.meta')
        logging.debug('Restoring model')
        saver.restore(self._session, checkpoint_name)

    def _restore_network(self, restore_from: str, restore_model_name: Optional[str]=None) -> None:
        """
        Restore TF net from the given checkpoint.
        :param restore_from: path to directory from which the model is restored
        :param restore_model_name: model name to be restored (e.g. `model.ckpt`)
        """

        logging.info('Restoring model from `{}`'.format(restore_from))
        assert path.isdir(restore_from), '`BaseTFNet` expect `restore_from` to be an existing directory.'
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
                                 'lacks `net.restore_model_name`. Please, specify it.'.format(restore_from))

            logging.info('Restoring model from checkpoint `{}` located in directory `{}`'.format(restore_model_name,
                                                                                                 restore_from))
            self._restore_checkpoint(path.join(restore_from, restore_model_name))

    @property
    def restore_fallback_module(self) -> str:
        return 'cxflow_tf'

    @property
    def restore_fallback_class(self) -> str:
        return 'BaseTFNet'

    def _create_net(self, **kwargs) -> None:
        """
        Create network according to the given config.

        -------------------------------------------------------
        cxflow framework requires the following
        -------------------------------------------------------
        1. define training op named as 'train_op'
        2. input/output tensors have to be named according to net.io config
        3. initialize variables through self._session
        -------------------------------------------------------

        :param kwargs: net configuration
        """
        raise NotImplementedError('`_create_net` method must be implemented in order to construct a new network.')
