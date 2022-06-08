from cmath import log
import tensorflow as tf
import datetime

class ScalarWriter:
    def __init__(self, log_dir = None):
        """ key : list of str, the name of variable to be saved """

        if log_dir is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = 'logs/scalar/' + current_time
       
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.steps = dict()

    def record(self, key, value, step = None):
        if not key in self.steps.keys():
            self.steps[key] = 1

        with self.writer.as_default():
            rec_step = self.steps[key] if step is None else step
            tf.summary.scalar(key, value, step = rec_step)
        
        self.steps[key] = self.steps[key] + 1
         
