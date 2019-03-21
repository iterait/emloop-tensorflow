"""
Test module for base TensorFlow models (:py:class:`emloop_tensorflow.BaseModel`).
"""

from os import path

import numpy as np
import pytest

from emloop import MainLoop
from emloop.tests.main_loop_test import SimpleDataset
from emloop.hooks import StopAfter
from emloop_tensorflow import FrozenModel

from .model_test import TrainableModel, _OPTIMIZER, _IO


def test_frozen_model_restore(tmpdir):
    """Test frozen model restoration."""
    with pytest.raises(ValueError):
        FrozenModel(inputs=[], outputs=[], restore_from=tmpdir)  # there is no .pb file yet

    dummy_model = TrainableModel(dataset=None, log_dir=tmpdir, **_IO, freeze=True, optimizer=_OPTIMIZER)
    dummy_model.save('')

    # restore from directory
    FrozenModel(**_IO, restore_from=tmpdir)

    # restore from file
    FrozenModel(**_IO, restore_from=path.join(tmpdir, 'model.pb'))

    # wrong configurations
    dummy_model.save('another')
    with pytest.raises(ValueError):
        FrozenModel(**_IO, restore_from=tmpdir)  # multiple .pb files

    with pytest.raises(ValueError):
        FrozenModel(**_IO, restore_from='/something/that/does/not/exist')


def test_frozen_model_misc(tmpdir):
    """Test various frozen model attributes."""
    dummy_model = TrainableModel(dataset=None, log_dir=tmpdir, **_IO, freeze=True, optimizer=_OPTIMIZER)
    dummy_model.save('')

    # restore from directory
    frozen_model = FrozenModel(**_IO, restore_from=tmpdir, session_config={'allow_soft_placement': True})

    assert frozen_model.input_names == _IO['inputs']
    assert frozen_model.output_names == _IO['outputs']

    with pytest.raises(NotImplementedError):
        frozen_model.save('fail')


def test_frozen_model_run(tmpdir):
    """Test frozen model run after restoration."""
    # train and freeze a model
    dataset = SimpleDataset()
    model = TrainableModel(dataset=dataset, log_dir=tmpdir, **_IO, freeze=True, optimizer=_OPTIMIZER)
    mainloop = MainLoop(model=model, dataset=dataset, hooks=[StopAfter(epochs=1000)], skip_zeroth_epoch=False)
    mainloop.run_training()
    model.save('')

    frozen_model = FrozenModel(inputs=['input'], outputs=['output'], restore_from=tmpdir)

    with pytest.raises(AssertionError):
        frozen_model.run({}, True, None)

    outputs = frozen_model.run({'input': [[1]*10]})
    assert np.allclose(outputs['output'][0], [0]*10, atol=0.001)
