import os
import tensorflow as tf
from datetime import datetime

# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
from dataset import Dataset
from cycle_gan import cycleGAN


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = Dataset(self.flags.dataset, self.flags)
        self.model = cycleGAN(self.sess, self.flags, self.dataset.image_size, self.dataset())

        self._make_folders()

        self.sess.run(tf.global_variables_initializer())
        tf_utils.show_all_variables()

    def _make_folders(self):
        if self.flags.is_train:  # train stage
            cur_time = datetime.now().strftime("%Y%m%d-%H%M")
            self.model_out_dir = "{}/model_{}".format(self.flags.dataset, cur_time)
            if not os.path.isdir(self.model_out_dir):
                os.makedirs(self.model_out_dir)

            self.sample_out_dir = "{}/sample_{}".format(self.flags.dataset, cur_time)
            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

        elif not self.flags.is_train:  # test stage
            self.model_out_dir = "{}/model_{}".format(self.flags.dataset, self.flags.load_model)

            self.test_out_dir = "{}/test_{}".format(self.flags.dataset, self.flags.load_model)
            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

    @staticmethod
    def train():
        print('hello train!')

    @staticmethod
    def test():
        print('hello test!')
