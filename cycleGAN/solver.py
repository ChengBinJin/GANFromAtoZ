# ---------------------------------------------------------
# Tensorflow CycleGAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import time
# import cv2
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
        self.iter_time = 0

        self._make_folders()

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        tf_utils.show_all_variables()

    def _make_folders(self):
        if self.flags.is_train:  # train stage
            if self.flags.load_model is None:
                cur_time = datetime.now().strftime("%Y%m%d-%H%M")
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)
                if not os.path.isdir(self.model_out_dir):
                    os.makedirs(self.model_out_dir)
            else:
                cur_time = self.flags.load_model
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)

            self.sample_out_dir = "{}/sample/{}".format(self.flags.dataset, cur_time)
            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

            self.train_writer = tf.summary.FileWriter("{}/logs/{}".format(self.flags.dataset, cur_time))

        elif not self.flags.is_train:  # test stage
            self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)

            self.test_out_dir = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

    def train(self):
        # load initialized checkpoint that provided
        if self.flags.load_model is not None:
            if self.load_model():
                print(' [*] Load SUCCESS!\n')
            else:
                print(' [!] Load Failed...\n')

        # threads for tfrecord
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            # for iter_time in range(self.flags.iters):
            while self.iter_time < self.flags.iters:
                # samppling images and save them
                self.sample()

                # train_step
                loss, summary = self.model.train_step()
                self.print_info(loss)
                self.train_writer.add_summary(summary, self.iter_time)
                self.train_writer.flush()

                # save model
                self.save_model()

                self.iter_time += 1

        except KeyboardInterrupt:
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # when done, ask the threads to stop
            coord.request_stop()
            coord.join(threads)

    def test(self):
        if self.load_model():
            print(' [*] Load SUCCESS!')
        else:
            print(' [!] Load Failed...')

        # read test data
        test_data_files = utils.all_files_under(self.dataset.night_path)
        total_time = 0.

        for idx in range(len(test_data_files)):
            img = utils.imagefiles2arrs([test_data_files[idx]])  # read img
            img = utils.transform(img)  # convert [0, 255] to [-1., 1.]

            # measure inference time
            start_time = time.time()
            imgs = self.model.test_step(img, mode='YtoX')  # inference
            total_time += time.time() - start_time

            self.model.plots(imgs, idx, self.dataset.image_size, self.test_out_dir)  # write results

        print('Avg PT: {:3f} msec.'.format(total_time / len(test_data_files) * 1000.))

    def sample(self):
        if np.mod(self.iter_time, self.flags.sample_freq) == 0:
            imgs = self.model.sample_imgs()
            self.model.plots(imgs, self.iter_time, self.dataset.image_size, self.sample_out_dir)

    def print_info(self, loss):
        if np.mod(self.iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('G_loss', loss[0]), ('Dy_loss', loss[1]),
                                                  ('F_loss', loss[2]), ('Dx_loss', loss[3]),
                                                  ('dataset', self.dataset.name),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(self.iter_time, ord_output)

    def save_model(self):
        if np.mod(self.iter_time + 1, self.flags.save_freq) == 0:
            model_name = 'model'
            self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name),
                            global_step=self.iter_time)

            print('=====================================')
            print('             Model saved!            ')
            print('=====================================\n')

    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            print('===========================')
            print('   iter_time: {}'.format(self.iter_time))
            print('===========================')

            return True
        else:
            return False
