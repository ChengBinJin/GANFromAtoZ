# ---------------------------------------------------------
# Tensorflow VAE Implementation for Day2Night Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow_utils as tf_utils
import utils as utils

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


# noinspection PyPep8Naming
class VAE(object):
    def __init__(self, sess, flags, image_size, log_path=None):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size

        self.output_dim = self.image_size[0] * self.image_size[1] * self.image_size[2]
        self.log_path = log_path
        self.n_hidden = 500

        self._init_logger()     # init logger
        self._build_net()       # init graph
        self._tensorboard()       # init tensorboard
        logger.info('Initialized VAE SUCCESS!')

    def _init_logger(self):
        if self.flags.is_train:
            tf_utils._init_logger(self.log_path)

    def _build_net(self):
        self.x_hat_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='input_img')
        self.x_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='target_img')
        self.keep_prob_tfph = tf.placeholder(tf.float32, name='keep_prob')
        self.z_in_tfph = tf.placeholder(tf.float32, shape=[None, self.flags.z_dim], name='latent_variable')

        # encoding
        mu, sigma = self.encoder(self.x_hat_tfph)
        # sampling by re-parameterization technique
        self.z = mu + sigma * tf.random_normal(tf.shape(mu), mean=0., stddev=1., dtype=tf.float32)

        # decoding
        y = self.decoder(self.z)
        self.y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)
        sample_y = self.decoder(self.z_in_tfph, is_reuse=True)
        self.sample_y = tf.clip_by_value(sample_y, 1e-8, 1 - 1e-8)

        # loss
        marginal_likelihood = tf.reduce_sum(self.x_tfph * tf.log(self.y) + (1 - self.x_tfph) * tf.log(1 - self.y),
                                            [1, 2, 3])
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

        self.marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        self.KL_divergence = tf.reduce_mean(KL_divergence)
        self.ELBO = self.marginal_likelihood - self.KL_divergence
        self.loss = - self.ELBO

        self.train_op = tf.train.AdamOptimizer(self.flags.learning_rate).minimize(self.loss)

    def _tensorboard(self):
        tf.summary.scalar('loss/marginal_likelihood', -self.marginal_likelihood)
        tf.summary.scalar('loss/KL_divergence', self.KL_divergence)
        tf.summary.scalar('loss/total_loss', self.loss)

        self.summary_op = tf.summary.merge_all()

    def encoder(self, data, name='encoder'):
        with tf.variable_scope(name):
            data_flatten = flatten(data)
            tf_utils.print_activations(data_flatten)

            # 1st hidden layer
            h0_linear = tf_utils.linear(data_flatten, self.n_hidden, name='h0_linear')
            h0_elu = tf_utils.elu(h0_linear, name='h0_elu')
            h0_drop = tf.nn.dropout(h0_elu, keep_prob=self.keep_prob_tfph, name='h0_drop')
            tf_utils.print_activations(h0_drop)

            # 2nd hidden layer
            h1_linear = tf_utils.linear(h0_drop, self.n_hidden, name='h1_linear')
            h1_tanh = tf_utils.tanh(h1_linear, name='h1_tanh')
            h1_drop = tf.nn.dropout(h1_tanh, keep_prob=self.keep_prob_tfph, name='h1_drop')
            tf_utils.print_activations(h1_drop)

            # 3rd hidden layer
            h2_linear = tf_utils.linear(h1_drop, 2*self.flags.z_dim, name='h2_linear')
            tf_utils.print_activations(h2_linear)

            # The mean parameter is unconstrained
            mean = h2_linear[:, :self.flags.z_dim]
            # The standard deviation must be positive.
            # Parameterize with a softplus and add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(h2_linear[:, self.flags.z_dim:])

            tf_utils.print_activations(mean)
            tf_utils.print_activations(stddev)

        return mean, stddev

    def decoder(self, z, name='decoder', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()
            tf_utils.print_activations(z)

            # 1st hidden layer
            h0_linear = tf_utils.linear(z, self.n_hidden, name='h0_linear')
            h0_tanh = tf_utils.tanh(h0_linear, name='h0_tanh')
            h0_drop = tf.nn.dropout(h0_tanh, keep_prob=self.keep_prob_tfph, name='h0_drop')
            tf_utils.print_activations(h0_drop)

            # 2nd hidden layer
            h1_linear = tf_utils.linear(h0_drop, self.n_hidden, name='h1_linear')
            h1_elu = tf_utils.elu(h1_linear, name='h1_elu')
            h1_drop = tf.nn.dropout(h1_elu, keep_prob=self.keep_prob_tfph, name='h1_drop')
            tf_utils.print_activations(h1_drop)

            # 3rd hidden layer
            h2_linear = tf_utils.linear(h1_drop, self.output_dim, name='h2_linear')
            h2_sigmoid = tf_utils.sigmoid(h2_linear, name='h2_sigmoid')
            tf_utils.print_activations(h2_sigmoid)

            output = tf.reshape(h2_sigmoid, [-1, *self.image_size])
            tf_utils.print_activations(output)

        return output

    def train_step(self, x_data, y_data):
        train_ops = [self.train_op, self.loss, self.KL_divergence, self.marginal_likelihood, self.summary_op]
        train_feed = {self.x_hat_tfph: x_data, self.x_tfph: y_data, self.keep_prob_tfph: 0.9}
        _, loss, KL_divergence, marginal_likelihood, summary = self.sess.run(train_ops, feed_dict=train_feed)

        return [loss, KL_divergence, -marginal_likelihood], summary

    def sample_imgs(self, x_data, y_data):
        test_feed = {self.x_hat_tfph: x_data, self.keep_prob_tfph: 1.0}
        y_fakes = self.sess.run(self.y, feed_dict=test_feed)

        return [x_data, y_fakes, y_data]

    def decoder_y(self, z):
        y_imgs = self.sess.run(self.sample_y, feed_dict={self.z_in_tfph: z, self.keep_prob_tfph: 1.0})
        return y_imgs

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('total_loss', loss[0]), ('KL_divergence', loss[1]),
                                                  ('marginal_likelihood', loss[2]), ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    def plots(self, imgs, iter_time, save_file):
        # parameters for plot size
        scale, margin = 0.015, 0.02
        n_cols, n_rows = len(imgs), imgs[0].shape[0]
        cell_size_h, cell_size_w = imgs[0].shape[1] * scale, imgs[0].shape[2] * scale

        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        # VAE output range is [0., 1.]
        # imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]

        # save more bigger image
        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                if self.image_size[2] == 3:
                    plt.imshow((imgs[col_index][row_index]).reshape(self.image_size), cmap='Greys_r')
                elif self.image_size[2] == 1:
                    plt.imshow((imgs[col_index][row_index]).reshape(self.image_size[:2]), cmap='Greys_r')

        plt.savefig(os.path.join(save_file, '{}.png').format(str(iter_time).zfill(7)), bbox_inches='tight')
        plt.close(fig)
