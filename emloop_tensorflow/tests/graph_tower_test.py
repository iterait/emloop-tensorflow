"""
Test module for ``GraphTower`` class (:py:class:`emloop_tensorflow.GraphTower`).
"""

import pytest
import tensorflow as tf

from emloop_tensorflow.graph_tower import GraphTower


def test_find_io_tensors():
    """Test finding IO tensors in the ``GraphTower``."""
    tower = GraphTower(id_=-1, inputs=['x'], outputs=['y', 'loss'], loss_name='loss')
    with tower:
        x = tf.placeholder(tf.float32, (10, 13), name='x')
        y = tf.identity(x * 10, name='y')
        tf.reduce_mean(y*0, axis=1, name='loss')
    tower.find_io_tensors()
    assert tower['x'] == x
    with pytest.raises(KeyError):
        tower['undefined']
