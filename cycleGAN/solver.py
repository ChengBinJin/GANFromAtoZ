import os
import cv2
import collections
import numpy as np
import tensorflow as tf
from datetime import datetime

# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
import utils as utils
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

    def train(self):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            for iter_time in range(self.flags.iters):
                self.sample(iter_time)  # sampling images and save them

                loss = self.model.train_step()
                self.print_info(iter_time, loss)

        except KeyboardInterrupt:
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # when done, ask the threads to stop
            coord.request_stop()
            coord.join(threads)

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            imgs = self.model.sample_imgs()
            utils.plots(imgs, iter_time, self.dataset.image_size, self.sample_out_dir)

    def print_info(self, iter_time, loss):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('G_loss', loss[0]), ('Dy_loss', loss[1]),
                                                  ('F_loss', loss[2]), ('Dx_loss', loss[3]),
                                                  ('dataset', self.dataset.name),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)


    @staticmethod
    def test():
        print('hello solver TEST function!')
