# ---------------------------------------------------------
# Tensorflow VAE Implementation for Day2Night Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf
from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('batch_size', 32, 'batch size, default: 32')
tf.flags.DEFINE_string('dataset', 'paired', 'dataset name, default: paired')
tf.flags.DEFINE_integer('which_direction', 0, 'AtoB (0) or BtoA (1), default: AtoB 0')

tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate for Adam, default: 0.001')
tf.flags.DEFINE_integer('z_dim', 128, 'dimension of z vector, default: 128')

tf.flags.DEFINE_integer('iters', 20, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 1, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 10, 'save frequency for model, default: 10000')
tf.flags.DEFINE_integer('sample_freq', 5, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_integer('sample_batch', 8, 'number of sampling images for check generator quality, default: 8')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model taht you wish to continue training '
                       '(e.g. 20181108-1029), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
    if not FLAGS.is_train:
        solver.test()


if __name__ == '__main__':
    tf.app.run()
