"""
Module with learning rate decaying hook.
"""
import logging

import tensorflow as tf
from cxflow.hooks.abstract_hook import AbstractHook
from cxflow_tensorflow import BaseModel


class DecayLR(AbstractHook):
    """
    Hook for modifying (decaying) the learning rate (or any other variable) during the training.

    It expects a variable with the specified name to be present in the TF model being trained.

    After every epoch, the variable is either multiplied or summed with the specified ``decay_value``.


    .. code-block:: yaml
        :caption: multiply ``learning_rate`` variable by 0.98 after every epoch

        hooks:
          - cxflow_tensorflow.DecayLR


    .. code-block:: yaml
        :caption: linear decay of ``my_learning_rate`` variable

        hooks:
          - cxflow_tensorflow.DecayLR:
              decay_value=-0.00001
              variable_name='my_learning_rate'
              decay_type='multiply'

    """

    LR_DECAY_TYPES = {'multiply', 'add'}
    """Possible LR decay types."""

    def __init__(self, model: BaseModel, decay_value: int=0.98, variable_name: str='learning_rate',
                 decay_type: str='multiply', **kwargs):
        """
        Create new DecayLR hook.
        
        :param model: TF model being trained
        :param decay_value: the value to modify the learning rate with
        :param variable_name: variable name to be modified
        :param decay_type: decay type, one of :py:attr:`LR_DECAY_TYPES`
        """
        if not isinstance(model, BaseModel):
            raise TypeError('Invalid model class `{}`. '
                            'Only to models derived from '
                            '`cxflow_tensorflow.BaseModel` are allowed. '.format(type(model)))

        if decay_type not in DecayLR.LR_DECAY_TYPES:
            raise ValueError('Unrecognized LR decay type `{}`. '
                             'Allowed values are `{}`'.format(decay_type, DecayLR.LR_DECAY_TYPES))

        if decay_type == 'multiply' and decay_value <= 0:
            raise ValueError('Invalid lr decay value `{}` for multiply lr decay. '
                             'Only positive values are allowed for multiply LR decay.'.format(decay_value))

        self._decay_value = decay_value
        self._decay_type = decay_type
        self._model = model
        self._lr = self._model.graph.get_tensor_by_name(variable_name+':0')

        super().__init__(**kwargs)

    def after_epoch(self, **_) -> None:
        """
        Modify the specified TF variable (now saved in ``self._lr``) using ``self._decay_value``.

        :param _: ignore all the parameters
        """
        old_value = self._lr.eval(session=self._model.session)
        new_value = old_value * self._decay_value if self._decay_type == 'multiply' else old_value + self._decay_value

        tf.assign(self._lr,  new_value).eval(session=self._model.session)
        logging.info('LR updated to `%s`', self._lr.eval(session=self._model.session))
