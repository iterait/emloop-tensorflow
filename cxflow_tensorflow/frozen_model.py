import logging
from os import path
from abc import ABCMeta
from typing import List, Mapping, Optional
from glob import glob

import cxflow as cx
import tensorflow as tf

from .graph_tower import GraphTower
from .model import BaseModel


class FrozenModel(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 inputs: List[str], outputs: List[str],
                 dataset: Optional[cx.AbstractDataset]=None, log_dir: Optional[str]=None,
                 session_config: Optional[dict]=None, n_gpus: int=0, restore_from: Optional[str]=None, **_):
        assert 0 <= n_gpus <= 1, 'FrozenModel can be used only with n_gpus=0 or n_gpus=1'
        super().__init__(dataset=dataset, log_dir=log_dir, restore_from=restore_from)

        self._dataset = dataset
        self._log_dir = log_dir

        self._graph = None
        self._tower = GraphTower(n_gpus-1, inputs, outputs, None)  # -1 == CPU tower

        logging.info('\tCreating TF model on %s GPU devices', n_gpus)
        self._graph = tf.Graph()
        self._session = self._create_session(session_config)

        with self._graph.as_default():
            self._restore_model(restore_from)
            self._tower.find_io_tensors()

    def run(self, batch: cx.Batch, train: bool=False, stream: cx.datasets.StreamWrapper=None) -> Mapping[str, object]:
        """
        Run the model with the given ``batch``. Update the trainable variables only if ``train`` is true.

        Fetch and return all the model outputs as a dict.

        :param batch: batch dict ``{source_name: values}``
        :param train: flag whether parameters update (``train_op``) should be included in fetches
        :param stream: stream wrapper (useful for precise buffer management)
        :raise ValueError: if an output is wrongly typed or its batch size differs from the input batch size
        :return: outputs dict
        """
        feed_dict = {}
        fetches = []

        for placeholder_name in self.input_names:
            feed_dict[self._tower[placeholder_name]] = batch[placeholder_name]
        for output_name in self.output_names:
            fetches.append(self._tower[output_name])

        if stream is not None:
            with stream.allow_buffering:
                outputs = self._session.run(fetches=fetches, feed_dict=feed_dict)
        else:
            outputs = self._session.run(fetches=fetches, feed_dict=feed_dict)

        return dict(zip(self.output_names, outputs))

    def save(self, name_suffix: str='') -> str:
        raise NotImplementedError('Frozen model cannot be saved.')

    def _restore_model(self, restore_from: str) -> None:
        """
        Restore TF model from the given ``restore_from`` path.

        The model name can be derived if the ``restore_from`` is a directory containing exactly one checkpoint or if
        its base name specifies a checkpoint.

        :param restore_from: path to directory from which the model is restored, optionally with model name as the last
        part
        """
        logging.info('Restoring model from `{}`'.format(restore_from))
        restore_model_name = None
        if not path.isdir(restore_from):
            restore_model_name = path.basename(restore_from)
            restore_from = path.dirname(restore_from)
        assert path.isdir(restore_from), '`FrozenModel` expects `restore_from` to be an existing directory.'

        # attempt to derive the model name
        if restore_model_name is None:
            pb_files = glob('{}/*.pb'.format(restore_from))
            if len(pb_files) == 0:
                raise ValueError('No `{}/*.pb` files found.'.format(restore_from))
            elif len(pb_files) == 1:
                logging.info('Restoring model from frozen graph`{}`'.format(pb_files[0]))
                restore_model_name = path.basename(pb_files[0])
            else:
                raise ValueError('There are multiple frozen graph files found in the directory {}. '
                                 'Please specify the full frozen graph path.'.format(restore_from))

        logging.info('Restoring model from frozen graph `{}` located in directory `{}`'
                     .format(restore_model_name, restore_from))
        checkpoint_path = path.join(restore_from, restore_model_name)
        # restore the graph
        with tf.gfile.GFile(checkpoint_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    @property
    def restore_fallback(self) -> str:
        return 'cxflow_tensorflow.FrozenModel'
