import tensorflow as tf
import cv2
import scipy.misc

# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
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
        self.ngf = 32

        self._G_gen_train_ops, self._F_gen_train_ops = [], []

        self._build_net()

    def _build_net(self):
        # tfph: TensorFlow PlaceHolder
        self.x_input_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='x_input_tfph')
        self.y_input_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='y_input_tfph')

        self.G_gen = Generator(name='G', ngf=self.ngf, norm=self.norm, image_size=self.image_size,
                               _ops=self._G_gen_train_ops)
        self.Dy_dis = Discriminator(name='Dy')
        self.F_gen = Generator(name='F', ngf=self.ngf, norm=self.norm, image_size=self.image_size)
        self.Dx_dis = Discriminator(name='Dx')

        x_reader = Reader(self.x_path, name='X', image_size=self.image_size, batch_size=self.flags.batch_size)
        y_reader = Reader(self.y_path, name='Y', image_size=self.image_size, batch_size=self.flags.batch_size)
        x_imgs = x_reader.feed()
        y_imgs = y_reader.feed()

        outut = self.G_gen(x_imgs)

    def cycle_consistency_loss(self, x_imgs, y_imgs):
        print('hello cycle_consistency loss!')

    def _tensorboard(self):
        print('hello _tensorboard!')

    # def generator(self):
    #     print('hello generator!')
    #
    # def discriminator(self):
    #     print('hello discriminator!')

    def train_step(self):
        print('hello train_step!')

    def write_tensorboard(self):
        print('hello write_tensorboard!')


class Generator(object):
    def __init__(self, name=None, ngf=32, norm='instance', image_size=(128, 256, 3), _ops=None):
        self.name = name
        self.ngf = ngf
        self.norm = norm
        self.image_size = image_size
        self._ops = _ops
        self.reuse = False

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H, W, 32)
            conv1 = tf_utils.padding2d(x, p_h=3, p_w=3, pad_type='REFLECT', name='conv1_padding')
            conv1 = tf_utils.conv2d(conv1, self.ngf, k_h=7, k_w=7, d_h=1, d_w=1, padding='VALID',
                                    name='conv1_conv')
            conv1 = tf_utils.norm(conv1, _type='instance', _ops=self._ops, name='conv1_norm')
            conv1 = tf_utils.relu(conv1, name='conv1_relu', is_print=True)

            # (N, H, W, 32)  -> (N, H/2, W/2, 64)
            conv2 = tf_utils.conv2d(conv1, 2*self.ngf, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm',)
            conv2 = tf_utils.relu(conv2, name='conv2_relu', is_print=True)

            # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
            conv3 = tf_utils.conv2d(conv2, 4*self.ngf, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm',)
            conv3 = tf_utils.relu(conv3, name='conv3_relu', is_print=True)

            # (N, H/4, W/4, 128) -> (N, H/4, W/4, 128)
            if (self.image_size[0] <= 128) and (self.image_size[1] <= 128):
                # use 6 residual blocks for 128x128 images
                res_out = tf_utils.n_res_blocks(conv3, num_blocks=6, is_print=True)
            else:
                # use 9 blocks for higher resolution
                res_out = tf_utils.n_res_blocks(conv3, num_blocks=9, is_print=True)

            # (N, H/4, W/4, 128) -> (N, H/2, W/2, 64)
            conv4 = tf_utils.deconv2d(res_out, 2*self.ngf, name='conv4_deconv2d')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.relu(conv4, name='conv4_relu', is_print=True)

            # (N, H/2, W/2, 64) -> (N, H, W, 32)
            conv5 = tf_utils.deconv2d(conv4, self.ngf, name='conv5_deconv2d')
            conv5 = tf_utils.norm(conv5, _type='instance', _ops=self._ops, name='conv5_norm')
            conv5 = tf_utils.relu(conv5, name='conv5_relu', is_print=True)

            # (N, H, W, 32) -> (N, H, W, 3)
            output = tf_utils.padding2d(conv5, p_h=3, p_w=3, pad_type='REFLECT', name='output_padding')
            output = tf_utils.conv2d(output, self.image_size[2], k_h=7, k_w=7, d_h=1, d_w=1,
                                     padding='VALID', name='output_conv')
            output = tf_utils.tanh(output, name='output_tanh', is_print=True)

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            return output


class Discriminator(object):
    def __init__(self, name=''):
        self.name = name
        print('hello {} discriminator!'.format(self.name))
