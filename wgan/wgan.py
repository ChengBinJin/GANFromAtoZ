# ---------------------------------------------------------
# TensorFlow WGAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
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


class WGAN(object):
    def __init__(self, sess, flags, image_size, data_path):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size

        self.x_path, self.y_path = data_path[0], data_path[1]
        self._gen_train_ops, self._dis_train_ops = [], []
        self.gen_c = [1024, 512, 256, 128, 64, 64]  # 4, 8, 16, 32, 64, 128
        self.dis_c = [64, 128, 256, 512, 512, 512]  # 128, 64, 32, 16, 8, 4

        if self.flags.is_train:
            self._build_net()
            self._tensorboard()
        else:
            self._build_net(is_train=False)
        print("Initialized WGAN SUCCESS!")

    def _build_net(self, is_train=True):
        if is_train is True:
            self.z = tf.placeholder(tf.float32, shape=[None, self.flags.z_dim], name='latent_vector')
            y_reader = Reader(self.y_path, name='Y', image_size=self.image_size, batch_size=self.flags.batch_size)
            self.y_imgs = y_reader.feed()

            self.g_samples = self.generator(self.z)
            _, d_logit_real = self.discriminator(self.y_imgs)
            _, d_logit_fake = self.discriminator(self.g_samples, is_reuse=True)

            # discriminator loss
            self.d_loss = tf.reduce_mean(d_logit_real) - tf.reduce_mean(d_logit_fake)
            # generator loss
            self.g_loss = -tf.reduce_mean(d_logit_fake)

            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')

            # Optimizers for generator and discriminator
            dis_op = tf.train.RMSPropOptimizer(learning_rate=self.flags.learning_rate).minimize(
                -self.d_loss, var_list=d_vars)
            dis_ops = [dis_op] + self._dis_train_ops
            self.dis_optim = tf.group(*dis_ops)
            self.clip_dis = [var.assign(tf.clip_by_value(var, -self.flags.clip_val, self.flags.clip_val))
                             for var in d_vars]

            gen_op = tf.train.RMSPropOptimizer(learning_rate=self.flags.learning_rate).minimize(
                self.g_loss, var_list=g_vars)
            gen_ops = [gen_op] + self._gen_train_ops
            self.gen_optim = tf.group(*gen_ops)
        else:
            self.z = tf.placeholder(tf.float32, shape=[None, self.flags.z_dim], name='latent_vector')
            self.g_samples = self.generator(self.z)

    def _tensorboard(self):
        tf.summary.scalar('loss/d_loss', self.d_loss)
        tf.summary.scalar('loss/g_loss', self.g_loss)

        self.summary_op = tf.summary.merge_all()

    def generator(self, data, name='g_'):
        with tf.variable_scope(name):
            data_flatten = flatten(data)

            # 2 x 4
            h0_linear = tf_utils.linear(data_flatten, 2*4*self.gen_c[0], name='h0_linear')
            h0_reshape = tf.reshape(h0_linear, [tf.shape(h0_linear)[0], 2, 4, self.gen_c[0]])
            h0_batchnorm = tf_utils.batch_norm(h0_reshape, name='h0_batchnorm', _ops=self._gen_train_ops)
            h0_relu = tf.nn.relu(h0_batchnorm, name='h0_relu')

            # 4 x 8
            h1_deconv = tf_utils.deconv2d(h0_relu, self.gen_c[1], name='h1_deconv2d')
            h1_batchnorm = tf_utils.batch_norm(h1_deconv, name='h1_batchnorm', _ops=self._gen_train_ops)
            h1_relu = tf.nn.relu(h1_batchnorm, name='h1_relu')

            # 8 x 16
            h2_deconv = tf_utils.deconv2d(h1_relu, self.gen_c[2], name='h2_deconv2d')
            h2_batchnorm = tf_utils.batch_norm(h2_deconv, name='h2_batchnorm', _ops=self._gen_train_ops)
            h2_relu = tf.nn.relu(h2_batchnorm, name='h2_relu')

            # 16 x 32
            h3_deconv = tf_utils.deconv2d(h2_relu, self.gen_c[3], name='h3_deconv2d')
            h3_batchnorm = tf_utils.batch_norm(h3_deconv, name='h3_batchnorm', _ops=self._gen_train_ops)
            h3_relu = tf.nn.relu(h3_batchnorm, name='h3_relu')

            # 32 x 64
            h4_deconv = tf_utils.deconv2d(h3_relu, self.gen_c[4], name='h4_deconv2d')
            h4_batchnorm = tf_utils.batch_norm(h4_deconv, name='h4_batchnorm', _ops=self._gen_train_ops)
            h4_relu = tf.nn.relu(h4_batchnorm, name='h4_relu')

            # 64 x 128
            h5_deconv = tf_utils.deconv2d(h4_relu, self.gen_c[5], name='h5_deconv2d')
            h5_batchnorm = tf_utils.batch_norm(h5_deconv, name='h5_batchnorm', _ops=self._gen_train_ops)
            h5_relu = tf.nn.relu(h5_batchnorm, name='h5_relu')

            # 128 x 256
            output = tf_utils.deconv2d(h5_relu, self.image_size[2], name='h6_deconv2d')
            return tf.nn.tanh(output)

    def discriminator(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            # (128, 256) -> (64, 128)
            h0_conv = tf_utils.conv2d(data, self.dis_c[0], name='h0_conv2d')
            h0_lrelu = tf_utils.lrelu(h0_conv, name='h0_lrelu')

            # (64, 128) -> (32, 64)
            h1_conv = tf_utils.conv2d(h0_lrelu, self.dis_c[1], name='h1_conv2d')
            h1_batchnorm = tf_utils.batch_norm(h1_conv, name='h1_batchnorm', _ops=self._dis_train_ops)
            h1_lrelu = tf_utils.lrelu(h1_batchnorm, name='h1_lrelu')

            # (32, 64) -> (16, 32)
            h2_conv = tf_utils.conv2d(h1_lrelu, self.dis_c[2], name='h2_conv2d')
            h2_batchnorm = tf_utils.batch_norm(h2_conv, name='h2_batchnorm', _ops=self._dis_train_ops)
            h2_lrelu = tf_utils.lrelu(h2_batchnorm, name='h2_lrelu')

            # (16, 32) -> (8, 16)
            h3_conv = tf_utils.conv2d(h2_lrelu, self.dis_c[3], name='h3_conv2d')
            h3_batchnorm = tf_utils.batch_norm(h3_conv, name='h3_batchnorm', _ops=self._dis_train_ops)
            h3_lrelu = tf_utils.lrelu(h3_batchnorm, name='h3_lrelu')

            # (8, 16) -> (4, 8)
            h4_conv = tf_utils.conv2d(h3_lrelu, self.dis_c[4], name='h4_conv2d')
            h4_batchnorm = tf_utils.batch_norm(h4_conv, name='h4_batchnorm', _ops=self._dis_train_ops)
            h4_lrelu = tf_utils.lrelu(h4_batchnorm, name='h4_lrelu')

            # (4, 8) -> (2, 4)
            h5_conv = tf_utils.conv2d(h4_lrelu, self.dis_c[5], name='h5_conv2d')
            h5_batchnorm = tf_utils.batch_norm(h5_conv, name='h5_batchnorm', _ops=self._dis_train_ops)
            h5_lrelu = tf_utils.lrelu(h5_batchnorm, name='h5_lrelu')

            h5_flatten = flatten(h5_lrelu)
            h6_linear = tf_utils.linear(h5_flatten, 1, name='h6_linear')

            return tf.nn.sigmoid(h6_linear), h6_linear

    def train_step(self):
        d_loss, summary_d, summary_g = None, None, None

        # train discriminator
        for idx in range(self.flags.num_critic):
            dis_feed = {self.z: self.sample_z(num=self.flags.batch_size)}
            _, d_loss = self.sess.run([self.dis_optim, self.d_loss], feed_dict=dis_feed)
            self.sess.run(self.clip_dis)

        # train generator
        gen_feed = {self.z: self.sample_z(num=self.flags.batch_size)}
        _, g_loss, summary = self.sess.run([self.gen_optim, self.g_loss, self.summary_op], feed_dict=gen_feed)

        return [d_loss, g_loss], summary

    def sample_imgs(self):
        g_feed = {self.z: self.sample_z(num=self.flags.sample_batch)}
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
                                                  ('d_loss', loss[0]), ('g_loss', loss[1]),
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
        n_cols, n_rows = int(np.sqrt(len(imgs) / 2)), int(np.sqrt(len(imgs) / 2)) * 2
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
