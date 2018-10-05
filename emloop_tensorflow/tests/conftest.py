import pytest
import tensorflow as tf
import numpy as np

from emloop_tensorflow.hooks import DecayLR, DecayLROnPlateau
from emloop.hooks import OnPlateau


class CV2Mock:
    def imencode(*args, **kwargs):
        return None, np.array([10, 20, 30], dtype=np.uint8)


@pytest.fixture
def mock_to_writer(mocker):
    yield mocker.patch('tensorflow.summary.FileWriter', autospec=True)


@pytest.fixture
def mock_object_writer(mocker):
    yield mocker.patch.object(tf.summary.FileWriter, 'add_summary')


@pytest.fixture
def mock_dict_writer(mocker):
    yield mocker.patch.dict('sys.modules', **{'cv2': CV2Mock})


@pytest.fixture
def mock_object_decaylr_wait(mocker):
    yield mocker.patch.object(DecayLROnPlateau, '_decay_variable')


@pytest.fixture
def mock_object_decaylr(mocker):
    yield mocker.patch.object(DecayLR, 'after_epoch')


@pytest.fixture
def mock_object_onplateau(mocker):
    yield mocker.patch.object(OnPlateau, 'after_epoch')
