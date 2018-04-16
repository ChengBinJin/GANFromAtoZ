import tensorflow as tf
import cv2
import scipy.misc
from reader import Reader


# noinspection PyPep8Naming
class cycleGAN(object):
    def __init__(self, sess, flags, image_size, data_path):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size
        self.x_path, self.y_path = data_path[0], data_path[1]

        # True: use lsgan (mean squared error)
        # False: use cross entropy loss
        self.use_lsgan = True
        self.use_sigmoid = not self.use_lsgan
        # [instance|batch] use instance norm or batch norm, default: instance
        self.norm = 'instane'
        self.lambda1, self.lambda2 = 10.0, 10.0
        self.ngf = 64

        self._build_net()

    def _build_net(self):
        x_reader = Reader(self.x_path, name='X', image_size=self.image_size, batch_size=self.flags.batch_size)
        y_reader = Reader(self.y_path, name='Y', image_size=self.image_size, batch_size=self.flags.batch_size)
        x_imgs = x_reader.feed()
        y_imgs = y_reader.feed()


    def _tensorboard(self):
        print('hello _tensorboard!')

    def generator(self, name):
        print('hello generator!')

    def discriminator(self):
        print('hello discriminator!')

    def train_step(self):
        print('hello train_step!')

    def write_tensorboard(self):
        print('hello write_tensorboard!')
