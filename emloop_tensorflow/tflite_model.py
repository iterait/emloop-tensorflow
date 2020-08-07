import logging
from os import path
from glob import glob
from typing import List, Mapping, Optional

import emloop as el
import tensorflow as tf

from .graph_tower import GraphTower
from .model import BaseModel
from .utils import Profiler
import numpy as np

class TFLiteModel(el.AbstractModel):
    """
    :py:class:`TFLiteModel` is **emloop** compatible abstraction for loading and running TFLite graphs
    (.tflite files).

    In order to use it, just change the ``model.class`` configuration and invoke any **emloop** command such as
    ``emloop eval ...``.

    .. code-block:: yaml
        :caption: using TFLite model

        # ...
        model:
          class: emloop_tensorflow.TFLiteModel
          # ...

    """

    def __init__(self, inputs, outputs, restore_from: str, dataset, log_dir: Optional[str]=None, n_gpus=0,
                 quantize_stats:Optional[dict] = None, **_):
        """
        Initialize new :py:class:`TFLiteModel` instance.

        :param log_dir: output directory
        :param restore_from: restore model path (either a dir or a .tflite file)
        """
        self.quantize_stats = quantize_stats
        super().__init__(None, '', restore_from)
        self._interpreter, self._inputs, self._outputs = self.restore_tflite_model(restore_from, inputs, outputs)

    def run(self, batch: el.Batch, train: bool=False, stream: el.datasets.StreamWrapper=None) -> Mapping[str, object]:
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
        assert not train, 'TFModel cannot be trained.'

        for input_name, input_detail in self._inputs.items():
            if input_name not in batch:
                raise ValueError(f'Missing {input_name} in the input batch')
            if self.quantize_stats and input_name in self.quantize_stats:
                input_data = np.uint8(batch[input_name])
            else:
                input_data = np.float32(batch[input_name])
            self._interpreter.set_tensor(input_detail['index'], input_data)

        self._interpreter.invoke()

        def dequant(tensor, tensor_name):
            if self.quantize_stats and tensor_name in self.quantize_stats:
                return self.quantize_stats[tensor_name]['scale']*\
                       (np.float32(tensor)+self.quantize_stats[tensor_name]['shift'])
            else:
                return tensor

        results = {output_name: dequant(self._interpreter.get_tensor(output_detail['index']), output_name)
                   for output_name, output_detail in self._outputs.items()}
        return results

    def save(self, name_suffix: str = '') -> str:
        """Save the model (not implemented)."""
        raise NotImplementedError('TFLite model cannot be saved.')

    @property
    def input_names(self) -> List[str]:  # pylint: disable=invalid-sequence-index
        """List of TF input tensor (placeholder) names."""
        return self._inputs.keys()

    @property
    def output_names(self) -> List[str]:  # pylint: disable=invalid-sequence-index
        """List of TF output tensor names."""
        return self._outputs.keys()

    @staticmethod
    def restore_tflite_model(restore_from: str, input_names, output_names) -> None:
        """
        Restore TFLite model from the given ``restore_from`` path.

        The model name can be derived if the ``restore_from`` is a directory containing exactly one checkpoint or if
        its base name specifies a checkpoint.

        :param restore_from: path to directory from which the model is restored, optionally including model filename
        """
        logging.info('Restoring TFLite model from `{}`'.format(restore_from))
        restore_model_name = None
        if not path.isdir(restore_from):
            restore_model_name = path.basename(restore_from)
            restore_from = path.dirname(restore_from)
        if not path.isdir(restore_from):
            raise ValueError('TFLite model restore path `{}` is not a directory.'.format(restore_from))

        # attempt to derive the model name
        if restore_model_name is None:
            tflite_files = glob('{}/*.tflite'.format(restore_from))
            if len(tflite_files) == 0:
                raise ValueError('No `{}/*.tflite` files found.'.format(restore_from))
            elif len(tflite_files) == 1:
                logging.info('Restoring model from TFLite model`{}`'.format(tflite_files[0]))
                restore_model_name = path.basename(tflite_files[0])
            else:
                raise ValueError('There are multiple TFLite model files found in the directory {}. '
                                 'Please specify the full TFLite model path.'.format(restore_from))

        logging.info('Restoring model from TFLite model `{}` located in directory `{}`'
                     .format(restore_model_name, restore_from))
        model_path = path.join(restore_from, restore_model_name)

        # restore the graph
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        inputs = {input_name: input_detail
                       for input_name, input_detail in zip(input_names, interpreter.get_input_details())}
        outputs = {output_name: output_detail
                        for output_name, output_detail in zip(output_names, interpreter.get_output_details())}
        return interpreter, inputs, outputs
