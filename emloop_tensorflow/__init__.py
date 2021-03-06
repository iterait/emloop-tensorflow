"""
Main emloop-tensorflow module exposing the :py:class:`emloop_tensorflow.BaseModel` allowing to define **emloop** trainable models (networks).

Additional hooks, ops and util functions are available in the respective sub-modules.

The main design goal is to **allow focusing on the model architecture** while most of the burden code is hidden to the user.
In fact, in most cases one will override only a single method :py:meth:`emloop_tensorflow.BaseModel._create_model`.
"""
from .model import BaseModel
from .frozen_model import FrozenModel
from .tflite_model import TFLiteModel
from .utils import create_activation, create_optimizer
from .metrics import bin_dice, bin_stats

from . import hooks
from . import ops
from . import utils
from . import metrics
from . import models

__all__ = ['BaseModel', 'FrozenModel', 'TFLiteModel']

__version__ = '0.6.0'
