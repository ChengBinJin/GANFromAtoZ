# ---------------------------------------------------------
# TensorFlow WGAN-GP Implementation for Day2Night
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import logging
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from reader import Reader
import tensorflow_utils as tf_utils
import utils as utils

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


# noinspection PyPep8Naming
class WGAN_GP(object):
    def __init__(self, sess, flags, image_size, data_path, log_path=None):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size
        self.log_path = log_path

        self.x_path, self.y_path = data_path[0], data_path[1]
        self.gen_train_ops, self.dis_train_ops = [], []
        self.gen_c = [2*4*512, 512, 256, 128, 64, 32, 32]
        self.dis_c = [64, 128, 256, 512, 512, 512, 512]

        self._init_logger()  # init logger
        self._build_net()  # init graph
        if self.flags.is_train:
            self._tensorboard()  # init tensorboard

        logger.info("Initialized WGAN-GP SUCCESS!")

    def _init_logger(self):
        if self.flags.is_train:
            tf_utils._init_logger(self.log_path)

    def _build_net(self):
        self.z = tf.placeholder(tf.float32, shape=[None, self.flags.z_dim], name='latent_vector')

        if self.flags.is_train:
            y_reader = Reader(self.y_path, name='Y', image_size=self.image_size, batch_size=self.flags.batch_size)
            self.y_imgs = y_reader.feed()

            self.g_samples = self.generator(self.z)
            _, d_logit_real = self.discriminator(self.y_imgs)
            _, d_logit_fake = self.discriminator(self.g_samples, is_reuse=True)

            # discriminator loss
            self.wgan_d_loss = tf.reduce_mean(d_logit_fake) - tf.reduce_mean(d_logit_real)
            # generator loss
            self.g_loss = -tf.reduce_mean(d_logit_fake)

            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')

            # gradient penalty
            self.gp_loss = self.gradient_penalty()
            self.d_loss = self.wgan_d_loss + self.flags.lambda_ * self.gp_loss

            # Optimizers for generator and discriminator
            self.gen_optim = tf.train.AdamOptimizer(
                learning_rate=self.flags.learning_rate, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_vars)
            self.dis_optim = tf.train.AdamOptimizer(
                learning_rate=self.flags.learning_rate, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_vars)
        else:
            self.g_samples = self.generator(self.z)

    def gradient_penalty(self):
        alpha = tf.random_uniform(shape=[self.flags.batch_size, 1, 1, 1], minval=0., maxval=1.)
        differences = self.g_samples - self.y_imgs
        interpolates = self.y_imgs + (alpha * differences)
        gradients = tf.gradients(self.discriminator(interpolates, is_reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return gradient_penalty

    def _tensorboard(self):
        tf.summary.scalar('loss/negative_wgan_d_loss', -self.wgan_d_loss)
        tf.summary.scalar('loss/gp_loss', self.gp_loss)
        tf.summary.scalar('loss/total_negative_d_loss', -self.d_loss)  # negative critic loss
        tf.summary.scalar('loss/g_loss', self.g_loss)

        self.summary_op = tf.summary.merge_all()

    def generator(self, data, name='g_'):
        with tf.variable_scope(name):
            data_flatten = flatten(data)
            tf_utils.print_activations(data_flatten)

            # from (N, 128) to (N, 2, 4, 512)
            h0_linear = tf_utils.linear(data_flatten, self.gen_c[0], name='h0_linear')
            h0_reshape = tf.reshape(h0_linear, [tf.shape(h0_linear)[0], 2, 4, int(self.gen_c[0]/(2*4))])

            # (N, 4, 8, 512)
            resblock_1 = tf_utils.res_block_v2(h0_reshape, self.gen_c[1], filter_size=3, _ops=self.gen_train_ops,
                                               norm_='batch', resample='up', name='res_block_1')
            # (N, 8, 16, 256)
            resblock_2 = tf_utils.res_block_v2(resblock_1, self.gen_c[2], filter_size=3, _ops=self.gen_train_ops,
                                               norm_='batch', resample='up', name='res_block_2')
            # (N, 16, 32, 128)
            resblock_3 = tf_utils.res_block_v2(resblock_2, self.gen_c[3], filter_size=3, _ops=self.gen_train_ops,
                                               norm_='batch', resample='up', name='res_block_3')
            # (N, 32, 64, 64)
            resblock_4 = tf_utils.res_block_v2(resblock_3, self.gen_c[4], filter_size=3, _ops=self.gen_train_ops,
                                               norm_='batch', resample='up', name='res_block_4')
            # (N, 64, 128, 32)
            resblock_5 = tf_utils.res_block_v2(resblock_4, self.gen_c[5], filter_size=3, _ops=self.gen_train_ops,
                                               norm_='batch', resample='up', name='res_block_5')
            # (N, 128, 256, 32)
            resblock_6 = tf_utils.res_block_v2(resblock_5, self.gen_c[6], filter_size=3, _ops=self.gen_train_ops,
                                               norm_='batch', resample='up', name='res_block_6')

            norm_7 = tf_utils.norm(resblock_6, _type='batch', _ops=self.gen_train_ops, name='norm_7')
            relu_7 = tf_utils.relu(norm_7, name='relu_7')

            # (N, 128, 256, 3)
            output = tf_utils.conv2d(relu_7, output_dim=self.image_size[2], k_w=3, k_h=3, d_h=1, d_w=1, name='output')

            return tf_utils.tanh(output)

    def discriminator(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()
            tf_utils.print_activations(data)

            # (N, 128, 256, 64)
            conv_0 = tf_utils.conv2d(data, output_dim=self.dis_c[0], k_h=3, k_w=3, d_h=1, d_w=1, name='conv_0')
            # (N, 64, 128, 128)
            resblock_1 = tf_utils.res_block_v2(conv_0, self.dis_c[1], filter_size=3, _ops=self.dis_train_ops,
                                               norm_='layer', resample='down', name='res_block_1')
            # (N, 32, 64, 256)
            resblock_2 = tf_utils.res_block_v2(resblock_1, self.dis_c[2], filter_size=3, _ops=self.dis_train_ops,
                                               norm_='layer', resample='down', name='res_block_2')
            # (N, 16, 32, 512)
            resblock_3 = tf_utils.res_block_v2(resblock_2, self.dis_c[3], filter_size=3, _ops=self.dis_train_ops,
                                               norm_='layer', resample='down', name='res_block_3')
            # (N, 8, 16, 512)
            resblock_4 = tf_utils.res_block_v2(resblock_3, self.dis_c[4], filter_size=3, _ops=self.dis_train_ops,
                                               norm_='layer', resample='down', name='res_block_4')
            # (N, 4, 8, 512)
            resblock_5 = tf_utils.res_block_v2(resblock_4, self.dis_c[5], filter_size=3, _ops=self.dis_train_ops,
                                               norm_='layer', resample='down', name='res_block_5')
            # (N, 2, 4, 512)
            resblock_6 = tf_utils.res_block_v2(resblock_5, self.dis_c[6], filter_size=3, _ops=self.dis_train_ops,
                                               norm_='layer', resample='down', name='res_block_6')

            # (N, 2*4*512)
            flatten_7 = flatten(resblock_6)
            output = tf_utils.linear(flatten_7, 1, name='output')

            return tf.nn.sigmoid(output), output

    def train_step(self):
        wgan_d_loss, gp_loss, d_loss = None, None, None

        # train discriminator
        for idx in range(self.flags.num_critic):
            dis_feed = {self.z: self.sample_z(num=self.flags.batch_size)}
            dis_run = [self.dis_optim, self.wgan_d_loss, self.gp_loss, self.d_loss]
            _, wgan_d_loss, gp_loss, d_loss = self.sess.run(dis_run, feed_dict=dis_feed)

        # train generator
        gen_feed = {self.z: self.sample_z(num=self.flags.batch_size)}
        _, g_loss, summary = self.sess.run([self.gen_optim, self.g_loss, self.summary_op], feed_dict=gen_feed)

        # negative critic loss
        return [-wgan_d_loss, gp_loss, -d_loss, g_loss], summary

    def sample_imgs(self, sample_size=64):
        g_feed = {self.z: self.sample_z(num=sample_size)}
        y_fakes, y_imgs = self.sess.run([self.g_samples, self.y_imgs], feed_dict=g_feed)

        return [y_fakes, y_imgs]

    def sample_test(self):
        g_feed = {self.z: self.sample_z(num=self.flags.sample_batch)}
        y_fakes = self.sess.run(self.g_samples, feed_dict=g_feed)

        return [y_fakes]

    def sample_z(self, num=64):
        return np.random.uniform(-1., 1., size=[num, self.flags.z_dim])

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('wgan_d_loss', loss[0]), ('gp_loss', loss[1]),
                                                  ('d_loss', loss[2]), ('g_loss', loss[3]),
                                                  ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    def plots(self, imgs_, iter_time, save_file):
        # reshape image from vector to (N, H, W, C)
        imgs_fake = np.reshape(imgs_[0], (self.flags.sample_batch, *self.image_size))
        imgs_real = np.reshape(imgs_[1], (self.flags.batch_size, *self.image_size))

        imgs = []
        for img in imgs_fake:
            imgs.append(img)
        for img in imgs_real[:self.flags.sample_batch]:
            imgs.append(img)

        # parameters for plot size
        scale, margin = 0.02, 0.01
        n_row = int(np.sqrt(len(imgs) / 2))
        n_cols = int((len(imgs) / 2) / n_row)
        n_rows = n_row * 2
        cell_size_h, cell_size_w = imgs[0].shape[0] * scale, imgs[0].shape[1] * scale

        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]

        # save more bigger image
        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                if self.image_size[2] == 3:
                    plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
                        self.image_size[0], self.image_size[1], self.image_size[2]), cmap='Greys_r')
                elif self.image_size[2] == 1:
                    plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
                        self.image_size[0], self.image_size[1]), cmap='Greys_r')
                else:
                    raise NotImplementedError

        plt.savefig(save_file + '/sample_{}.png'.format(str(iter_time)), bbox_inches='tight')
        plt.close(fig)

    def plots_test(self, imgs_, iter_time, save_file):
        # reshape image from vector to (N, H, W, C)
        imgs_fake = np.reshape(imgs_[0], (self.flags.sample_batch, *self.image_size))

        imgs = []
        for img in imgs_fake:
            imgs.append(img)

        # parameters for plot size
        scale, margin = 0.02, 0.01
        n_cols, n_rows = int(np.sqrt(len(imgs))), int(np.sqrt(len(imgs)))
        cell_size_h, cell_size_w = imgs[0].shape[0] * scale, imgs[0].shape[1] * scale

        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]

        # save more bigger image
        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
                    self.image_size[0], self.image_size[1], self.image_size[2]), cmap='Greys_r')

        plt.savefig(save_file + '/sample_{}.png'.format(str(iter_time)), bbox_inches='tight')
        plt.close(fig)
