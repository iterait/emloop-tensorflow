"""
Test module for utils module.
"""

import numpy as np
import tensorflow as tf

from cxflow_tensorflow.tests.test_core import CXTestCaseWithDir
from cxflow_tensorflow import repeat, create_optimizer


class UtilsTest(CXTestCaseWithDir):
    """
    Test case for utils helpers.
    """

    def test_repeat(self):
        """ Test if `repeat` works the same as np.repeat."""

        with tf.Session().as_default():
            # try different tensor types
            for npdtype, tfdtype in [(np.int32, tf.int32), (np.float32, tf.float32)]:
                for init_value in [np.array([0, 1, 2, 3], dtype=npdtype),
                                   np.array([[0, 1], [2, 3], [4, 5]], dtype=npdtype)]:
                    # and all their axes
                    for axis in range(len(init_value.shape)):
                        for repeats in [1, 2, 3, 11]:
                            tensor = tf.constant(init_value, dtype=tfdtype)

                            repeated_value = repeat(tensor, repeats=repeats, axis=axis).eval()
                            expected_value = np.repeat(init_value, repeats=repeats, axis=axis)

                            self.assertTrue(np.all(repeated_value == expected_value))

    def test_create_optimizer(self):
        """Test if create optimizer does work with tf optimizers."""

        optimizer_config = {'learning_rate': 0.1}

        # test missing required entry `class`
        self.assertRaises(AssertionError, create_optimizer, optimizer_config)

        optimizer_config['class'] = 'GradientDescentOptimizer'

        with tf.Session().as_default():
            # test if the optimizer is created correctlyW
            optimizer = create_optimizer(optimizer_config)
            self.assertIsInstance(optimizer, tf.train.GradientDescentOptimizer)

            # test if learning_rate variable is created with the correct value
            lr_tensor = tf.get_default_graph().get_tensor_by_name('learning_rate:0')
            tf.get_default_session().run(tf.global_variables_initializer())
            self.assertAlmostEqual(lr_tensor.eval(), 0.1)

        optimizer_config2 = {'learning_rate': 0.1, 'class': 'MomentumOptimizer'}

        # test missing required argument (momentum in this case)
        with tf.Graph().as_default():
            self.assertRaises(TypeError, create_optimizer, optimizer_config2)



