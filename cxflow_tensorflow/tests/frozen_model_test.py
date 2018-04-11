"""
Test module for base TensorFlow models (:py:class:`cxflow_tensorflow.BaseModel`).
"""
from os import path

import numpy as np

from cxflow import MainLoop
from cxflow.tests.main_loop_test import SimpleDataset
from cxflow.tests.test_core import CXTestCaseWithDir
from cxflow.hooks import StopAfter
from cxflow_tensorflow import FrozenModel

from .model_test import TrainableModel, _OPTIMIZER, _IO


class FrozenModelTest(CXTestCaseWithDir):
    """
    Test case for ``FrozenModel``.
    """

    def test_frozen_model_restore(self):
        """
        Test frozen model restoration.
        """
        with self.assertRaises(ValueError):
            FrozenModel(inputs=[], outputs=[], restore_from=self.tmpdir)  # there is no .pb file yet

        dummy_model = TrainableModel(dataset=None, log_dir=self.tmpdir, **_IO, freeze=True, optimizer=_OPTIMIZER)
        dummy_model.save('')

        # restore from directory
        FrozenModel(**_IO, restore_from=self.tmpdir)

        # restore from file
        FrozenModel(**_IO, restore_from=path.join(self.tmpdir, 'model.pb'))

        # wrong configurations
        dummy_model.save('another')
        with self.assertRaises(ValueError):
            FrozenModel(**_IO, restore_from=self.tmpdir)  # multiple .pb files

        with self.assertRaises(ValueError):
            FrozenModel(**_IO, restore_from='/something/that/does/not/exist')

    def test_frozen_model_misc(self):
        """
        Test various frozen model attributes.
        """
        dummy_model = TrainableModel(dataset=None, log_dir=self.tmpdir, **_IO, freeze=True, optimizer=_OPTIMIZER)
        dummy_model.save('')

        # restore from directory
        frozen_model = FrozenModel(**_IO, restore_from=self.tmpdir, session_config={'allow_soft_placement': True})

        self.assertEqual(frozen_model.restore_fallback, 'cxflow_tensorflow.FrozenModel')
        self.assertListEqual(frozen_model.input_names, _IO['inputs'])
        self.assertListEqual(frozen_model.output_names, _IO['outputs'])

        with self.assertRaises(NotImplementedError):
            frozen_model.save('fail')

    def test_frozen_model_run(self):
        """
        Test frozen model run after restoration.
        """
        # train and freeze a model
        dataset = SimpleDataset()
        model = TrainableModel(dataset=dataset, log_dir=self.tmpdir, **_IO, freeze=True, optimizer=_OPTIMIZER)
        mainloop = MainLoop(model=model, dataset=dataset, hooks=[StopAfter(epochs=1000)], skip_zeroth_epoch=False)
        mainloop.run_training(None)
        model.save('')

        frozen_model = FrozenModel(inputs=['input'], outputs=['output'], restore_from=self.tmpdir)

        with self.assertRaises(AssertionError):
            frozen_model.run({}, True, None)

        outputs = frozen_model.run({'input': [[1]*10]})
        self.assertTrue(np.allclose(outputs['output'][0], [0]*10, atol=0.001))
