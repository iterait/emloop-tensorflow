"""
Module with learning rate initializing hook.
"""
import logging

import tensorflow as tf
import cxflow as cx

from ..model import BaseModel


class InitLR(cx.AbstractHook):
    """
    Hook for initializing the learning rate (or any other variable) before the training.

    This is useful for setting up a resumed training.

    It expects a variable with the specified name to be present in the TF model being trained.

    .. code-block:: yaml
        :caption: initialize ``learning_rate`` variable to 0.001

        hooks:
          - cxflow_tensorflow.hooks.InitLR:
              value: 0.001

    .. code-block:: yaml
        :caption: initialize ``my_variable`` variable to 42

        hooks:
          - cxflow_tensorflow.hooks.InitLR:
              variable_name: my_variable
              value: 42

    """

    def __init__(self, model: BaseModel, value: float, variable_name: str='learning_rate', **kwargs):
        """
        Create new InitLR hook.

        :param model: TF model being trained
        :param value: desired variable value
        :param variable_name: variable name to be initialize
        """
        if not isinstance(model, BaseModel):
            raise TypeError('Invalid model class `{}`. '
                            'Only to models derived from '
                            '`cxflow_tensorflow.BaseModel` are allowed. '.format(type(model)))

        self._value = value
        self._model = model
        self._variable = self._model.graph.get_tensor_by_name(variable_name + ':0')

        super().__init__(**kwargs)

    def before_training(self) -> None:
        """Set ``self._variable`` to ``self._value``."""
        tf.assign(self._variable, self._value).eval(session=self._model.session)
        logging.info('LR updated to `%s`', self._variable.eval(session=self._model.session))
