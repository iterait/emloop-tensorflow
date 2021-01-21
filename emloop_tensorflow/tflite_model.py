import logging
from os import path
from glob import glob
from typing import List, Mapping, Optional

import numpy as np
import emloop as el
import tensorflow as tf

from .graph_tower import GraphTower
from .model import BaseModel
from .utils import Profiler


class QuantizedTFLiteModel(el.AbstractModel):
    """
    :py:class:`QuantizedTFLiteModel` is **emloop** compatible abstraction for loading and running TFLite graphs
    (.tflite files).

    Use this model class to infer UINT8 quantized .tflite model (not compiled for EdgeTpu).
    In order to use it, just change the ``model.class`` configuration and invoke any **emloop** command such as
    ``emloop eval ...``.

    .. code-block:: yaml
        :caption: using TFLite model

        # ...
        model:
          class: emloop_tensorflow.QuantizedTFLiteModel
          restore_from: <path-to-tflite-file>
          quantize_stats:
            input_0:
              scale: in0
              shift: in0
            output_0:
              scale: out0
              shift: out0
            output_1:
              scale: out1
              shift: out1
          # ...

    """

    def __init__(self, restore_from: str, dataset, log_dir: Optional[str]=None, n_gpus=0,
                 quantize_stats:Optional[dict] = None, **_):
        """
        Initialize new :py:class:`QuantizedTFLiteModel` instance.

        :param log_dir: output directory
        :param restore_from: restore model path (either a dir or a .tflite file)
        """
        self._quantize_stats = quantize_stats
        super().__init__(None, '', restore_from)
        self._interpreter, self._inputs, self._outputs = self.restore_tflite_model(restore_from)

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
            if self._quantize_stats and input_name in self._quantize_stats:
                input_data = np.uint8(batch[input_name])
            else:
                input_data = np.float32(batch[input_name])
            self._interpreter.set_tensor(input_detail['index'], input_data)

        self._interpreter.invoke()
        return {output_name: self._dequant(self._interpreter.get_tensor(output_detail['index']), output_name)
                for output_name, output_detail in self._outputs.items()}

    def save(self, name_suffix: str = '') -> str:
        """Save the model (not implemented)."""
        raise NotImplementedError('TFLite model cannot be saved.')

    def _dequant(self, tensor, tensor_name):
        """Dequantize model output"""
        if self._quantize_stats and tensor_name in self._quantize_stats:
            return self._quantize_stats[tensor_name]['scale']*\
                (np.float32(tensor) + self._quantize_stats[tensor_name]['shift'])
        else:
            return tensor

    @property
    def input_names(self) -> List[str]:  # pylint: disable=invalid-sequence-index
        """List of TF input tensor (placeholder) names."""
        return self._inputs.keys()

    @property
    def output_names(self) -> List[str]:  # pylint: disable=invalid-sequence-index
        """List of TF output tensor names."""
        return self._outputs.keys()

    @staticmethod
    def restore_tflite_model(restore_from: str) -> None:
        """
        Restore TFLite model from the given ``restore_from`` path.

        The model name can be derived if the ``restore_from`` is a directory containing exactly one checkpoint or if
        its base name specifies a checkpoint.

        :param restore_from: path to directory from which the model is restored, optionally including model filename
        """
        logging.info(f'Restoring TFLite model from {restore_from}')
        if restore_from.endswith('.tflite'):
            model_path = restore_from
        else:
            if not path.isdir(restore_from):
                restore_from = path.dirname(restore_from)
            tflite_files = glob(f'{restore_from}/*.tflite')
            if not tflite_files:
                raise ValueError(f'No `{restore_from}/*.tflite` files found.')
            elif len(tflite_files) == 1:
                model_path = tflite_files[0]
            else:
                raise ValueError(f'There are multiple TFLite model files found in the directory {restore_from}.'
                                 'Please specify the full TFLite model path.')
        logging.info(f'Restoring model from TFLite model `{model_path}`')

        # restore the graph
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        inputs = {input_detail['name']: input_detail
                  for input_detail in interpreter.get_input_details()}
        outputs = {output_detail['name']: output_detail
                   for output_detail in interpreter.get_output_details()}
        return interpreter, inputs, outputs
