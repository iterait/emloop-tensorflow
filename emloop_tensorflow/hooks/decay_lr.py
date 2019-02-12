"""
Module with learning rate decaying hook.
"""
import logging

import tensorflow as tf
import emloop as el
from emloop.hooks import EveryNEpoch

from ..model import BaseModel


class DecayLR(EveryNEpoch):
    """
    Hook for modifying (decaying) the learning rate (or any other variable) during the training.

    It expects a variable with the specified name to be present in the TF model being trained.

    Each `n_epochs`, the variable is either multiplied or summed with the specified ``decay_value``.


    .. code-block:: yaml
        :caption: multiply ``learning_rate`` variable by 0.98 after every epoch

        hooks:
          - emloop_tensorflow.hooks.DecayLR

    .. code-block:: yaml
        :caption: multiply ``learning_rate`` variable by 0.999 each 5th epoch

        hooks:
          - emloop_tensorflow.hooks.DecayLR:
              decay_value: 0.999
              n_epochs: 5

    .. code-block:: yaml
        :caption: linear decay of ``my_learning_rate`` variable

        hooks:
          - emloop_tensorflow.hooks.DecayLR:
              decay_value: -0.00001
              variable_name: my_learning_rate
              decay_type: add

    """

    LR_DECAY_TYPES = {'multiply', 'add'}
    """Possible LR decay types."""

    def __init__(self, model: BaseModel, decay_value: float=0.98, variable_name: str='learning_rate',
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
                            '`emloop_tensorflow.BaseModel` are allowed. '.format(type(model)))

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

    def _decay_variable(self) -> None:
        """
        Modify the specified TF variable (now saved in ``self._lr``) using ``self._decay_value``.

        :param _: ignore all the parameters
        """
        old_value = self._lr.eval(session=self._model.session)
        new_value = old_value * self._decay_value if self._decay_type == 'multiply' else old_value + self._decay_value

        tf.assign(self._lr,  new_value).eval(session=self._model.session)
        logging.info('LR updated to `%s`', self._lr.eval(session=self._model.session))

    def _after_n_epoch(self, epoch_id: int, **_) -> None:
        """Call :py:meth:`_decay_variable`."""
        self._decay_variable()


class DecayLROnPlateau(el.hooks.OnPlateau, DecayLR):
    """
    Decay learning rate on plateau.

    After decaying, LR may be decayed again only after additional ``long_term`` epochs.

    Shares args from both :py:class:`DecayLR` and :py:class:`el.hooks.OnPlateau`.

    .. code-block:: yaml
        :caption: multiply the learning rate by 0.1 when the mean of last 100 valid ``accuracy`` values is
                  greater than the mean of last 30 ``accuracy`` values.

        hooks:
          - emloop_tensorflow.hooks.DecayLROnPlateau:
              long_term: 100
              short_term: 30
              variable: accuracy
              objective: max

    .. code-block:: yaml
        :caption: decay LR by 0.01 when valid ``loss`` plateau is detected

        hooks:
          - emloop_tensorflow.hooks.DecayLROnPlateau:
              decay_value: 0.01

    """

    def __init__(self, decay_value: float=0.1, **kwargs):
        super().__init__(decay_value=decay_value, **kwargs)
        self._prevent_decay = 0

    def _on_plateau_action(self, **kwargs) -> None:
        """Call :py:meth:`_decay_variable`."""
        if self._prevent_decay == 0:
            logging.info('Plateau detected, decaying learning rate')
            self._decay_variable()
            self._prevent_decay = self._long_term

    def after_epoch(self, **kwargs) -> None:
        if self._prevent_decay > 0:
            self._prevent_decay -= 1
        el.hooks.OnPlateau.after_epoch(self, **kwargs)
