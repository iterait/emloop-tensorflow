"""
Main cxflow-tensorflow module exposing the :py:class:`cxflow_tensorflow.BaseModel` allowing to define **cxflow** trainable models (networks).

Additional hooks, ops and util functions are available in the respective sub-modules.

The main design goal is to **allow focusing on the model architecture** while most of the burden code is hidden to the user.
In fact, in most cases one will override only a single method :py:meth:`cxflow_tensorflow.BaseModel._create_model`.
"""
from .model import BaseModel
from .utils import create_activation, create_optimizer
from .metrics import bin_dice, bin_stats

__all__ = ['BaseModel']

__version__ = '0.3.2'
