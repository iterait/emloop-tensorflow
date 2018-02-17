"""
Test module for ``GraphTower`` class (:py:class:`cxflow_tensorflow.GraphTower`).
"""
from unittest import TestCase

import tensorflow as tf
from cxflow_tensorflow.graph_tower import GraphTower


class GraphTowerTest(TestCase):
    """Test case for ``GraphTower`` features not covered by ``BaseModel`` tests."""

    def test_find_io_tensors(self):
        """Test finding IO tensors in the ``GraphTower``."""
        tower = GraphTower(id_=-1, inputs=['x'], outputs=['y', 'loss'], loss_name='loss')
        with tower:
            x = tf.placeholder(tf.float32, (10, 13), name='x')
            y = tf.identity(x * 10, name='y')
            tf.reduce_mean(y*0, axis=1, name='loss')
        tower.find_io_tensors()
        self.assertEqual(tower['x'], x)
        with self.assertRaises(KeyError):
            tower['undefined']
