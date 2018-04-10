import logging
from os import path
from glob import glob
from typing import List, Mapping, Optional

import cxflow as cx
import tensorflow as tf

from .graph_tower import GraphTower
from .model import BaseModel


class FrozenModel(cx.AbstractModel):
    """
    :py:class:`FrozenModel` is **cxflow** compatible abstraction for loading and running frozen TensorFlow graphs
    (.pb files).

    In order to use it, just change the ``model.class`` configuration and invoke any **cxflow** command such as
    ``cxflow eval ...``.

    .. code-block:: yaml
        :caption: using frozen model
        # ...
        model:
          class: cxflow_tensorflow.FrozenModel
          # ...
    """

    def __init__(self,
                 inputs: List[str], outputs: List[str], restore_from: str,
                 session_config: Optional[dict]=None, n_gpus: int=0, **_):
        """
        Initialize new :py:class:`FrozenModel` instance.

        :param inputs: model input names
        :param outputs: model output names
        :param restore_from: restore model path (either a dir or a .pb file)
        :param session_config: TF session configuration dict
        :param n_gpus: number of GPUs to use (either 0 or 1)
        """
        super().__init__(None, '', restore_from)
        assert 0 <= n_gpus <= 1, 'FrozenModel can be used only with n_gpus=0 or n_gpus=1'

        self._graph = None
        self._tower = GraphTower(n_gpus-1, inputs, outputs, None)  # -1 == CPU tower

        logging.info('\tCreating TF model on %s GPU devices', n_gpus)
        self._graph = tf.Graph()
        if session_config:
            session_config = tf.ConfigProto(**session_config)
        self._session = tf.Session(graph=self._graph, config=session_config)

        with self._graph.as_default():
            self.restore_frozen_model(restore_from)
            self._tower.find_io_tensors()
            try:
                self._is_training = self._graph.get_tensor_by_name(BaseModel.TRAINING_FLAG_NAME + ':0')
            except KeyError:
                self._is_training = tf.placeholder(tf.bool, [], BaseModel.TRAINING_FLAG_NAME)

    def run(self, batch: cx.Batch, train: bool=False, stream: cx.datasets.StreamWrapper=None) -> Mapping[str, object]:
        """
        Run the model with the given ``batch``.

        Fetch and return all the model outputs as a dict.

        .. warning::
            :py:class:`FrozenModel` can not be trained.

        :param batch: batch dict ``{source_name: values}``
        :param train: flag whether parameters update (``train_op``) should be included in fetches
        :param stream: stream wrapper (useful for precise buffer management)
        :return: outputs dict
        """
        assert not train, 'Frozen model cannot be trained.'
        feed_dict = {self._is_training: False}
        fetches = []

        for placeholder_name in self.input_names:
            feed_dict[self._tower[placeholder_name]] = batch[placeholder_name]
        for output_name in self.output_names:
            fetches.append(self._tower[output_name])

        outputs = self._session.run(fetches=fetches, feed_dict=feed_dict)

        return dict(zip(self.output_names, outputs))

    def save(self, name_suffix: str='') -> str:
        """Save the model (not implemented)."""
        raise NotImplementedError('Frozen model cannot be saved.')

    @property
    def input_names(self) -> List[str]:  # pylint: disable=invalid-sequence-index
        """List of TF input tensor (placeholder) names."""
        return self._tower.input_names

    @property
    def output_names(self) -> List[str]:  # pylint: disable=invalid-sequence-index
        """List of TF output tensor names."""
        return self._tower.output_names

    @property
    def restore_fallback(self) -> str:
        return 'cxflow_tensorflow.FrozenModel'

    @staticmethod
    def restore_frozen_model(restore_from: str) -> None:
        """
        Restore frozen TF model from the given ``restore_from`` path.

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
        if not path.isdir(restore_from):
            raise ValueError('Frozen model restore path `{}` is not a directory.'.format(restore_from))

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
