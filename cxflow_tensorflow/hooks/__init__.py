"""
Module providing a small collection of TF cxflow hooks.

The hooks are expected to be created and used by **cxflow**. For additional info about the **cxflow** hooks system
please refer to the `cxflow tutorial <https://cxflow.org/tutorial.html#hooks>`_.
"""
from .decay_lr_hook import DecayLR
from .write_tensorboard_hook import WriteTensorboard

__all__ = ['WriteTensorboard', 'DecayLR']
