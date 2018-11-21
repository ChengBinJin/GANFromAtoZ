# ---------------------------------------------------------
# Tensorflow WGAN-GP Implementation for Day2Night
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf

from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_integer('batch_size', 8, 'batch size for one feed forwrad, default: 8')
tf.flags.DEFINE_string('dataset', 'day2night', 'dataset name, default: day2night')

tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate, default: 0.0001')
tf.flags.DEFINE_integer('num_critic', 5, 'the number of iterations of the critic per generator iteration, default: 5')
tf.flags.DEFINE_integer('z_dim', 128, 'dimension of z vector, default: 128')
tf.flags.DEFINE_float('lambda_', 10., 'gradient penalty lambda hyperparameter, default: 10.')
tf.flags.DEFINE_float('beta1', 0.5, 'beta1 momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('beta2', 0.9, 'beta2 momentum term of Adam, default: 0.9')

tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 10000, 'save frequency for model, default: 10000')
tf.flags.DEFINE_integer('sample_freq', 500, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_integer('sample_batch', 8, 'number of sampling images for check generator quality, default: 8')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to test, (e.g. 20180704-1736), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
    else:
        solver.test()


if __name__ == '__main__':
    tf.app.run()
