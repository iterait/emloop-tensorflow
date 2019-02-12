import tensorflow as tf
from tensorflow.python.client import timeline
from typing import Dict
import os


class Profiler:
    """
    Profiles tensorflow graphs and saves the profiles.
    """

    def __init__(self, log_dir: str, keep_profiles: int, session: tf.Session):
        """
        :param log_dir: directory where profiles will be saved
        :param keep_profiles: how many profiles are saved
        """
        self._log_dir = log_dir
        self._profile_counter = 0
        self._keep_profiles = keep_profiles
        self._run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self._session = session

    def run(self, fetches: Dict, feed_dict: Dict):
        """
        Evaluates the tensorflow graph with profiling, saves profile and returns outputs.

        :param session: tensorflow session
        :param fetches: names of output tensors
        :param feed_dict: input tensors
        """
        run_metadata = tf.RunMetadata()
        outputs = self._session.run(fetches=fetches, feed_dict=feed_dict,
                                    options=self._run_options, run_metadata=run_metadata)

        with open(os.path.join(self._log_dir, f'profile_{self._profile_counter}.json'), 'w') as ofile:
            tl = timeline.Timeline(run_metadata.step_stats)
            ofile.write(tl.generate_chrome_trace_format())

        self._profile_counter = (self._profile_counter + 1) % self._keep_profiles

        return outputs
