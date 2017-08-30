"""
Module with a tensorboard logging hook.
"""
import logging

import numpy as np
import tensorflow as tf

from cxflow.hooks import AbstractHook
from cxflow_tensorflow.model import BaseModel


class WriteTensorBoard(AbstractHook):
    """
    Write scalar epoch variables to TensorBoard summaries.

    Refer to TensorBoard `introduction <https://www.tensorflow.org/get_started/summaries_and_tensorboard>`_ for more
    info.

    By default, non-scalar values are ignored.

    .. code-block:: yaml
        :caption: default usage

        hooks:
          - cxflow_tensorflow.WriteTensorboard

    .. code-block:: yaml
        :caption: cast variables with unknown types to strings

        hooks:
          - cxflow_tensorflow.WriteTensorboard:
              on_unknown_type: str

    """

    UNKNOWN_TYPE_ACTIONS = {'error', 'warn', 'ignore'}
    """Possible actions to take on unknown variable type."""

    def __init__(self, model: BaseModel, output_dir: str, flush_secs: int=10, on_unknown_type: str='ignore', **kwargs):
        """
        Create new WriteTensorBoard hook.

        :param model: a BaseModel being trained
        :param output_dir: output dir to save the tensorboard logs
        :param on_unknown_type: an action to be taken if the variable value type is not supported (e.g. a list),
            one of :py:attr:`UNKNOWN_TYPE_ACTIONS`
        """
        assert isinstance(model, BaseModel)

        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self._on_unknown_type = on_unknown_type

        logging.debug('Creating TensorBoard writer')
        self._summary_writer = tf.summary.FileWriter(logdir=output_dir, graph=model.graph, flush_secs=flush_secs)

    def after_epoch(self, epoch_id: int, epoch_data: AbstractHook.EpochData) -> None:
        """
        Log the specified epoch data variables to the tensorboard.

        :param epoch_id: epoch ID
        :param epoch_data: epoch data as created by other hooks
        """
        logging.debug('TensorBoard logging after epoch %d', epoch_id)
        measures = []

        for stream_name in epoch_data.keys():
            stream_data = epoch_data[stream_name]
            for variable in stream_data.keys():
                value = stream_data[variable]
                if np.isscalar(value):  # try logging the scalar values
                    result = value
                elif isinstance(value, dict) and 'mean' in value:  # or the mean
                    result = value['mean']
                else:
                    err_message = 'Variable `{}` in stream `{}` is not scalar and does not contain `mean` aggregation'\
                                  .format(variable, stream_name)
                    if self._on_unknown_type == 'warn':
                        logging.warning(err_message)
                        result = str(value)
                    elif self._on_unknown_type == 'error':
                        raise ValueError(err_message)
                    else:
                        continue

                measures.append(tf.Summary.Value(tag='{}/{}'.format(stream_name, variable), simple_value=result))

        self._summary_writer.add_summary(tf.Summary(value=measures), epoch_id)
