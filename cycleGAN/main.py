import os
import tensorflow as tf
from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_string('dataset', 'day2night', 'dataset name, default; day2night')
tf.flags.DEFINE_bool('is_train', True, 'default: True')

tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial leraning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_integer('iters', 2000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 10, 'print frequency, default: 100')
tf.flags.DEFINE_integer('save_freq', 500, 'save frequency, default: 500')
tf.flags.DEFINE_integer('sample_freq', 20, 'sample frequency, default: 200')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20180412-1610), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
    if not FLAGS.is_train:
        solver.test()


if __name__ == '__main__':
    tf.app.run()
