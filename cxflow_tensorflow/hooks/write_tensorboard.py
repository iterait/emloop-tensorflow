"""
Module with a tensorboard logging hook.
"""
import logging
from typing import Optional, Iterable

import numpy as np
import tensorflow as tf
import cxflow as cx

from ..model import BaseModel


class WriteTensorBoard(cx.AbstractHook):
    """
    Write scalar epoch variables to TensorBoard summaries.

    Refer to TensorBoard `introduction <https://www.tensorflow.org/get_started/summaries_and_tensorboard>`_ for more
    info.

    By default, non-scalar values are ignored.

    .. code-block:: yaml
        :caption: default usage

        hooks:
          - cxflow_tensorflow.hooks.WriteTensorboard

    .. code-block:: yaml
        :caption: cast variables with unknown types to strings

        hooks:
          - cxflow_tensorflow.WriteTensorboard:
              on_unknown_type: str

    .. code-block:: yaml
        :caption: visualize the computational graph

        hooks:
          - cxflow_tensorflow.hooks.WriteTensorboard:
              visualize_graph: true

    """

    UNKNOWN_TYPE_ACTIONS = {'error', 'warn', 'ignore'}
    """Possible actions to take on unknown variable type."""

    MISSING_VARIABLE_ACTIONS = {'error', 'warn', 'ignore'}
    """Action executed on missing variable."""

    def __init__(self, model: BaseModel, output_dir: str, image_variables: Optional[Iterable[str]] = None,
                 flush_secs: int = 10, visualize_graph: bool = False, on_unknown_type: str = 'ignore',
                 on_missing_variable: str = 'error', **kwargs):
        """
        Create new WriteTensorBoard hook.

        :param model: a BaseModel being trained
        :param output_dir: output dir to save the tensorboard logs
        :param image_variables: list of image variable names
        :param flush_secs: flush interval in seconds
        :param visualize_graph: include visualization of the computational graph (may be resource-extensive)
        :param on_unknown_type: an action to be taken if the variable value type is not supported (e.g. a list),
            one of :py:attr:`UNKNOWN_TYPE_ACTIONS`
        :param on_missing_variable: an action to be taken if the specified variable is not present,
            one of :py:attr:`MISSING_VARIABLE_ACTIONS`
        """
        assert isinstance(model, BaseModel)
        assert on_unknown_type in WriteTensorBoard.UNKNOWN_TYPE_ACTIONS
        assert on_missing_variable in WriteTensorBoard.MISSING_VARIABLE_ACTIONS

        super().__init__(**kwargs)
        self._on_unknown_type = on_unknown_type
        self._image_variables = image_variables or []
        self._on_missing_variable = on_missing_variable

        logging.debug('Creating TensorBoard writer')
        graph = model.graph if visualize_graph else None
        self._summary_writer = tf.summary.FileWriter(logdir=output_dir, graph=graph, flush_secs=flush_secs)

    def after_epoch(self, epoch_id: int, epoch_data: cx.EpochData) -> None:
        """
        Log the specified epoch data variables to the tensorboard.

        :param epoch_id: epoch ID
        :param epoch_data: epoch data as created by other hooks
        """
        if self._image_variables:
            try:
                import cv2
            except (ImportError, ModuleNotFoundError) as ex:
                raise ImportError('OpenCV cv2 package is required for writing images to tensorboard.') from ex

        logging.debug('TensorBoard logging after epoch %d', epoch_id)
        summaries = []

        for stream_name in epoch_data.keys():
            stream_data = epoch_data[stream_name]
            for variable in stream_data.keys():  # try to treat all the variables but images as scalars
                if variable in self._image_variables:
                    continue  # we'll deal with image variables later
                value = stream_data[variable]
                result = None
                if np.isscalar(value):  # try logging the scalar values
                    result = value
                elif isinstance(value, dict) and 'mean' in value:  # or the mean
                    result = value['mean']
                elif isinstance(value, dict) and 'nanmean' in value:  # or the nanmean
                    result = value['nanmean']

                result_type = type(result)
                if np.issubdtype(result_type, float) or np.issubdtype(result_type, int):
                    summaries.append(tf.Summary.Value(tag='{}/{}'.format(stream_name, variable), simple_value=result))
                else:
                    err_message = 'Variable `{}` in stream `{}` has to be of type `int` or `float` ' \
                                  '(or a `dict` with a key named `mean` or `nanmean` whose corresponding value ' \
                                  'is of type `int` or `float`), found `{}` instead.'.format(variable, stream_name,
                                                                                             type(result))
                    if self._on_unknown_type == 'warn':
                        logging.warning(err_message)
                    elif self._on_unknown_type == 'error':
                        raise ValueError(err_message)

            for variable in self._image_variables:
                if variable not in stream_data:
                    err_message = '`{}` not found in epoch data.'.format(variable)
                    if self._on_missing_variable == 'error':
                        raise KeyError(err_message)
                    elif self._on_missing_variable == 'warn':
                        logging.warning(err_message)
                    continue
                image = stream_data[variable]
                assert isinstance(image, np.ndarray)
                assert image.ndim == 3 and image.shape[2] == 3
                if image.dtype in [np.float16, np.float32]:
                    image = ((image - np.min(image))*(255./(np.max(image)-np.min(image)))).astype(np.uint8)
                image = image.astype(np.uint8)
                image_string = cv2.imencode('.png', image)[1].tostring()

                image_summary = tf.Summary.Image(encoded_image_string=image_string,
                                                 height=image.shape[0],
                                                 width=image.shape[1])

                summaries.append(tf.Summary.Value(tag='{}/{}'.format(stream_name, variable), image=image_summary))

        self._summary_writer.add_summary(tf.Summary(value=summaries), epoch_id)
