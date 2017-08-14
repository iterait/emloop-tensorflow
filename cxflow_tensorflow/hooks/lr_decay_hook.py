"""
Module with learning rate decaying hook.
"""
import logging

import tensorflow as tf
from cxflow.hooks.abstract_hook import AbstractHook
from cxflow_tensorflow import BaseModel


class LRDecayHook(AbstractHook):
    """
    Hook for modifying (decaying) the learning rate (or any other variable) during the training.

    It expects an variable with the specified name to be present in the tf model being trained.

    After every epoch, the variable is either multiplied or summed with the specified decay_value.

    -------------------------------------------------------
    Example usage in config
    -------------------------------------------------------
    # for multiplying learning_rate variable by 0.98 after every epoch
    hooks:
      - class: LRDecayHook
    -------------------------------------------------------
    # for linear decay of my_learning_rate variable
    hooks:
      - class: LRDecayHook
        decay_value=-0.00001
        variable_name='my_learning_rate'
        decay_type='multiply'
    -------------------------------------------------------
    """

    LR_DECAY_TYPES = {'multiply', 'add'}

    def __init__(self, model: BaseModel,
                 decay_value=0.98, variable_name='learning_rate', decay_type='multiply', **kwargs):
        """
        Create new LRDecayHook.
        :param model: tf model being trained
        :param decay_value: the value to modify the learning rate
        :param variable_name: variable name to be modified
        :param decay_type: decay type, one of {'multiply', 'add'}
        """
        if not isinstance(model, BaseModel):
            raise TypeError('Invalid model class `{}`. '
                            'Only to models derived from '
                            '`cxflow_tensorflow.BaseModel` are allowed. '.format(type(model)))

        if decay_type not in LRDecayHook.LR_DECAY_TYPES:
            raise ValueError('Unrecognized LR decay type `{}`. '
                             'Allowed values are `{}`'.format(decay_type, LRDecayHook.LR_DECAY_TYPES))

        if decay_type == 'multiply' and decay_value <= 0:
            raise ValueError('Invalid lr decay value `{}` for multiply lr decay.'
                             'Only positive values are allowed for multiply LR decay.'.format(decay_value))

        self._decay_value = decay_value
        self._decay_type = decay_type
        self._model = model
        self._lr = self._model.graph.get_tensor_by_name(variable_name+':0')

        super().__init__(**kwargs)

    def after_epoch(self, **_) -> None:
        """
        Modify the specified tf variable (now saved in self._lr) with self._decay_value.
        """
        old_value = self._lr.eval(session=self._model.session)
        new_value = old_value * self._decay_value if self._decay_type == 'multiply' else old_value + self._decay_value

        tf.assign(self._lr,  new_value).eval(session=self._model.session)
        logging.info('LR updated to `%s`', self._lr.eval(session=self._model.session))
