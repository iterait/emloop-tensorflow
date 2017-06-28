"""
Test module for utils module.
"""

import numpy as np
import tensorflow as tf

from cxflow_tf.tests.test_core import CXTestCaseWithDir
from cxflow_tf import repeat


class UtilsTest(CXTestCaseWithDir):
    """
    Test case for utils helpers.
    """

    def test_repeat(self):
        """ Test if `repeat` works."""

        # try different tensor types
        for init_value in [np.array([0, 1, 2, 3], dtype=np.int32),
                           np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)]:
            # and all their axes
            for axis in range(len(init_value.shape)):
                # for some repeats (3 is often problematic)
                for repeats in [1, 2, 3, 11]:
                    tensor = tf.constant(init_value, dtype=tf.int32)
                    repeated_tensor = repeat(tensor, repeats=repeats, axis=axis)

                    sess = tf.Session()
                    repeated_value = sess.run([repeated_tensor])
                    expected_value = np.repeat(init_value, repeats=repeats, axis=axis)

                    # TODO: this must be deleted before merge
                    print(repeated_value)
                    print(expected_value)
                    print('OK' if np.all(repeated_value == expected_value) else 'Nope')
                    print('----')

                    self.assertTrue(np.all(repeated_value == expected_value))
