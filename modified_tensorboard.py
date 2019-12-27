"""
Adapted from:
https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
Updated for tensorflow 2
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Get rid of some tensorflow warnings

import time
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._log_write_dir = ''  # Need this or tf complains...
        self.step = 1

        logdir = self.get_run_logdir(os.path.join(os.curdir, 'my_logs'))
        self.writer = tf.summary.create_file_writer(logdir)

    def get_run_logdir(self, root_log_dir):
        run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
        return os.path.join(root_log_dir, run_id)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
            self.writer.flush()
