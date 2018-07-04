# ---------------------------------------------------------
# Tensorflow Vanilla GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import numpy as np
import tensorflow as tf
from datetime import datetime

from dataset import Dataset
from vanillaGAN import VanillaGAN
import tensorflow_utils as tf_utils


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = Dataset(self.flags.dataset, self.flags, image_size=(128, 256, 3))
        self.model = VanillaGAN(self.sess, self.flags, self.dataset.image_size, self.dataset())

        self._make_folders()

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        tf_utils.show_all_variables()

    def _make_folders(self):
        cur_time = datetime.now().strftime("%Y%m%d-%H%M")
        self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)
        if not os.path.isdir(self.model_out_dir):
            os.makedirs(self.model_out_dir)

        self.sample_out_dir = "{}/sample/{}".format(self.flags.dataset, cur_time)
        if not os.path.isdir(self.sample_out_dir):
            os.makedirs(self.sample_out_dir)

        self.train_writer = tf.summary.FileWriter("{}/logs/{}".format(self.flags.dataset, cur_time),
                                                  graph_def=self.sess.graph_def)

    def train(self):
        # threads for tfrecord
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            for iter_time in range(self.flags.iters):
                # samppling images and save them
                self.sample(iter_time)

                # train_step
                loss, summary = self.model.train_step()
                self.model.print_info(loss, iter_time)
                self.train_writer.add_summary(summary, iter_time)
                self.train_writer.flush()

                # save model
                self.save_model(iter_time)

        except KeyboardInterrupt:
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # when done, ask the thread to stop
            coord.request_stop()
            coord.join(threads)

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            imgs = self.model.sample_imgs()
            self.model.plots(imgs, iter_time, self.sample_out_dir)

    def save_model(self, iter_time):
        if np.mod(iter_time + 1, self.flags.save_freq) == 0:
            model_name = 'model'
            self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=iter_time)

            print('=====================================')
            print('             Model saved!            ')
            print('=====================================\n')
