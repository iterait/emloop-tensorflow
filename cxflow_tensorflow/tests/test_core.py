import tensorflow as tf

from cxflow.tests.test_core import CXTestCaseWithDir


class CXTestCaseWithDirAndModel(CXTestCaseWithDir):
    """Cxflow test case with temp dir and tf cleanup."""

    def __init__(self, *args, **kwargs):
        """Create a new test case."""
        super().__init__(*args, **kwargs)

    def tearDown(self):
        """Reset default tf graph after every test method."""
        tf.reset_default_graph()
        super().tearDown()
