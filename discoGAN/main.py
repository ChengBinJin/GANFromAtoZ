# ---------------------------------------------------------
# Tensorflow DiscoGAN Implementation for Day2Night Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf
from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('batch_size', 200, 'batch size, default: 200')
tf.flags.DEFINE_string('dataset', 'day2night', 'dataset name, default: day2night')
tf.flags.DEFINE_bool('is_train', False, 'training or inference mode, default: True')

tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'beta1 momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('beta2', 0.999, 'beta2 momentum term of Adam, default: 0.999')
tf.flags.DEFINE_float('weight_decay', 1e-4, 'hyper-parameter for regularization term')

tf.flags.DEFINE_integer('iters', 100000, 'number of iterations, default: 100000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 10000, 'save frequency for model, default: 10000')
tf.flags.DEFINE_integer('sample_freq', 500, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_integer('sample_batch', 200, 'number of sampling images for check generator quality, default: 200')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model taht you wish to continue training '
                       '(e.g. 20180907-1739), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
    if not FLAGS.is_train:
        solver.test()


if __name__ == '__main__':
    tf.app.run()
