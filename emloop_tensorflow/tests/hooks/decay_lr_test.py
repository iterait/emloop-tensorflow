"""
Test module for emloop_tensorflow.hooks.DecayLR hook.
"""
import pytest
import tensorflow as tf

from emloop_tensorflow.hooks import DecayLR, DecayLROnPlateau

from ..model_test import TrainableModel


class LRModel(TrainableModel):

    def _create_model(self, **kwargs):

        tf.Variable(2, name='learning_rate', dtype=tf.float32)

        super()._create_model(**kwargs)

    def _create_train_ops(self, *_):
        tf.no_op(name='train_op_1')


############
# Decay LR #
############
"""Test case for DecayLR."""


def test_invalid_config():
    """ Test ``DecayLR`` invalid configurations."""
    model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
    with pytest.raises(TypeError):
        DecayLR(model=42)
    with pytest.raises(ValueError):
        DecayLR(model, decay_value=-1)
    with pytest.raises(ValueError):
        DecayLR(model, decay_type='unrecognized')
    with pytest.raises(KeyError):
        DecayLR(model, variable_name='missing_variable')


def test_multiply():
    """ Test if ``DecayLR`` works properly in multiply mode."""
    decay_value = 0.9
    repeats = 13
    model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
    hook = DecayLR(model, decay_value=decay_value)
    hook.after_epoch(1)
    assert model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session) == pytest.approx(2*decay_value)
    for i in range(repeats):
        hook.after_epoch(i)
    assert model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session) == \
        pytest.approx(2 * (decay_value ** (1 + repeats)))


def test_multiply_n_epoch():
    """ Test if ``DecayLR`` works properly in multiply mode for every n epoch."""
    decay_value = 0.9
    repeats = 13
    n = 2
    model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
    hook = DecayLR(model, decay_value=decay_value, n_epochs=n)
    for i in range(repeats):
        hook.after_epoch(i)
    assert model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session) == \
        pytest.approx(2*(decay_value**(1+(repeats//n))))


def test_add():
    """ Test if ``DecayLR`` works properly in addition mode."""
    decay_value = 0.01
    repeats = 17
    model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
    hook = DecayLR(model, decay_value=decay_value, decay_type='add')
    hook.after_epoch(1)
    assert model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session) == pytest.approx(2+decay_value)
    for i in range(repeats):
        hook.after_epoch(i)
    assert model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session) == \
        pytest.approx(2+(decay_value*(1+repeats)))


def test_add_n_epoch():
    """ Test if ``DecayLR`` works properly in addition mode for every n epoch."""
    decay_value = 0.01
    repeats = 17
    n = 2
    model = LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])
    hook = DecayLR(model, decay_value=decay_value, decay_type='add', n_epochs=n)
    for i in range(repeats):
        hook.after_epoch(i)
    assert model.graph.get_tensor_by_name('learning_rate:0').eval(session=model.session) == \
        pytest.approx(2+(decay_value*(1+(repeats//n))))


#######################
# Decay LR On Plateau #
#######################
"""
Test case for DecayLROnPlateau.

Both OnPlateau and DecayLR are already tested, we only need to check if they are properly integrated.
"""


def get_model():
    return LRModel(dataset=None, log_dir='', inputs=['input', 'target'], outputs=['output'])


def get_epoch_data():
    return {'valid': {'loss': [0]}}


def test_arg_forward():
    """ Test if ``DecayLROnPlateau`` forwards args properly."""
    hook = DecayLROnPlateau(model=get_model(), short_term=5, variable='my_lr')
    assert hook._variable == 'my_lr'
    assert hook._short_term == 5


def test_call_forward(mock_object_decaylr, mock_object_onplateau):
    """ Test if ``DecayLROnPlateau`` forwards event calls properly."""
    hook = DecayLROnPlateau(model=get_model())
    hook.after_epoch(epoch_id=0, epoch_data=get_epoch_data())
    assert mock_object_decaylr.call_count == 0
    assert mock_object_onplateau.call_count == 1


def test_wait(mock_object_decaylr_wait):
    """ Test if ``DecayLROnPlateau`` waits short_term epochs between decays."""
    hook = DecayLROnPlateau(model=get_model(), long_term=4, short_term=3)
    hook._on_plateau_action()
    for i in range(hook._long_term):
        assert mock_object_decaylr_wait.call_count == 1
        hook.after_epoch(epoch_id=i, epoch_data=get_epoch_data())
        hook._on_plateau_action()
    assert mock_object_decaylr_wait.call_count == 2
