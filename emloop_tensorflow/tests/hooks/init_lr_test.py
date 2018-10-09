"""
Test module for emloop_tensorflow.hooks.InitLR hook.
"""
import pytest

from emloop_tensorflow.hooks import InitLR

from .decay_lr_test import LRModel


def test_creation():
    """ Test ``InitLR.__init__`` raises when configured improperly."""
    model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
    with pytest.raises(TypeError):
        InitLR(model=42, value=0.1)
    with pytest.raises(KeyError):
        InitLR(model=model, variable_name='does_not_exist', value=0.1)


def test_init():
    """Test ``InitLR`` hook initializes the variable properly."""
    init_value = 0.001

    model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
    hook = InitLR(model, value=init_value)

    lr = model.graph.get_tensor_by_name('learning_rate:0')
    assert round(abs(lr.eval(session=model.session)-2), 3) == 0
    hook.before_training()
    assert round(abs(lr.eval(session=model.session)-init_value), 3) == 0
