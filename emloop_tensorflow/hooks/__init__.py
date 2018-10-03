"""
Module providing a small collection of TF emloop hooks.

The hooks are expected to be created and used by **emloop**. For additional info about the **emloop** hooks system
please refer to the `emloop tutorial <https://emloop.org/tutorial.html#hooks>`_.
"""
from .decay_lr import DecayLR, DecayLROnPlateau
from .init_lr import InitLR
from .write_tensorboard import WriteTensorBoard

__all__ = ['WriteTensorBoard', 'DecayLR', 'InitLR', 'DecayLROnPlateau']
