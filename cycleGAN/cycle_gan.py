import tensorflow as tf
import cv2
import scipy.misc

# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
import utils as utils
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
        self.ngf, self.ndf = 32, 64
        self.real_label = 0.9
        self.start_dcay_step = 100000
        self.decay_steps = 100000
        self.eps = 1e-12

        self._G_gen_train_ops, self._F_gen_train_ops = [], []
        self._Dy_dis_train_ops, self._Dx_dis_train_ops = [], []

        self._build_net()

    def _build_net(self):
        # tfph: TensorFlow PlaceHolder
        self.x_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='x_test_tfph')
        self.y_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='y_test_tfph')

        self.G_gen = Generator(name='G', ngf=self.ngf, norm=self.norm, image_size=self.image_size,
                               _ops=self._G_gen_train_ops)
        self.Dy_dis = Discriminator(name='Dy', ndf=self.ndf, norm=self.norm, _ops=self._Dy_dis_train_ops)
        self.F_gen = Generator(name='F', ngf=self.ngf, norm=self.norm, image_size=self.image_size,
                               _ops=self._F_gen_train_ops)
        self.Dx_dis = Discriminator(name='Dx', ndf=self.ndf, norm=self.norm, _ops=self._Dx_dis_train_ops)

        x_reader = Reader(self.x_path, name='X', image_size=self.image_size, batch_size=self.flags.batch_size)
        y_reader = Reader(self.y_path, name='Y', image_size=self.image_size, batch_size=self.flags.batch_size)
        self.x_imgs = x_reader.feed()
        self.y_imgs = y_reader.feed()

        self.fake_x_pool_obj = utils.ImagePool(pool_size=50)
        self.fake_y_pool_obj = utils.ImagePool(pool_size=50)

        # cycle consistency loss
        cycle_loss = self.cycle_consistency_loss(self.x_imgs, self.y_imgs)

        # X -> Y
        self.fake_y_imgs = self.G_gen(self.x_imgs)
        G_gen_loss = self.generator_loss(self.Dy_dis, self.fake_y_imgs, use_lsgan=self.use_lsgan)
        G_loss = G_gen_loss + cycle_loss
        Dy_dis_loss = self.discriminator_loss(self.Dy_dis, self.y_imgs, self.fake_y_imgs,
                                              use_lsgan=self.use_lsgan)

        # Y -> X
        self.fake_x_imgs = self.F_gen(self.y_imgs)
        F_gen_loss = self.generator_loss(self.Dx_dis, self.fake_x_imgs, use_lsgan=self.use_lsgan)
        F_loss = F_gen_loss + cycle_loss
        Dx_dis_loss = self.discriminator_loss(self.Dx_dis, self.x_imgs, self.fake_x_imgs,
                                              use_lsgan=self.use_lsgan)

        G_optim = self.optimizer(loss=G_loss, variables=self.G_gen.variables, name='G_optim')
        Dy_optim = self.optimizer(loss=Dy_dis_loss, variables=self.Dy_dis.variables, name='Dy_optim')
        F_optim = self.optimizer(loss=F_loss, variables=self.F_gen.variables, name='F_optim')
        Dx_optim = self.optimizer(loss=Dx_dis_loss, variables=self.Dx_dis.variables, name='Dy_optim')
        self.optimizer = tf.group([G_optim, Dy_optim, F_optim, Dx_optim])

    def optimizer(self, loss, variables, name='optim'):
        global_step = tf.get_variable('global_step', 0, trainable=False)
        starter_learning_rate = self.flags.learning_rate
        end_learning_rate = 0.
        start_decay_step = self.start_dcay_step
        decay_steps = self.decay_steps

        learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                  tf.train.polynomial_decay(starter_learning_rate,
                                                            global_step - start_decay_step,
                                                            decay_steps, end_learning_rate),
                                  starter_learning_rate))

        learn_step = tf.train.AdamOptimizer(learning_rate, beta1=self.flags.beta1, name=name).\
            minimize(loss, global_step=global_step, var_list=variables)

        return learn_step

    def cycle_consistency_loss(self, x_imgs, y_imgs):
        forward_loss = tf.reduce_mean(tf.abs(self.F_gen(self.G_gen(x_imgs)) - x_imgs))
        backward_loss = tf.reduce_mean(tf.abs(self.G_gen(self.F_gen(y_imgs) - y_imgs)))
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss

    def generator_loss(self, dis_obj, fake_img, use_lsgan=True):
        if use_lsgan:
            # use mean squared error
            loss = tf.reduce_mean(tf.squared_difference(dis_obj(fake_img), self.real_label))
        else:
            # heuristic, non-saturating loss (I don't understand here!)
            # loss = -tf.reduce_mean(tf.log(dis_obj(fake_img) + self.eps)) / 2.  (???)
            loss = -tf.reduce_mean(tf.log(dis_obj(fake_img) + self.eps))
        return loss

    def discriminator_loss(self, dis_obj, real_img, fake_img, use_lsgan=True):
        if use_lsgan:
            # use mean squared error
            error_real = tf.reduce_mean(tf.squared_difference(dis_obj(real_img), self.real_label))
            error_fake = tf.reduce_mean(tf.square(dis_obj(fake_img)))
        else:
            # use cross entropy
            error_real = -tf.reduce_mean(tf.log(dis_obj(real_img) + self.eps))
            error_fake = -tf.reduce_mean(tf.log(1. - dis_obj(fake_img) + self.eps))

        # loss = (error_real + error_fake) / 2. (???)
        loss = error_real + error_fake
        return loss

    def _tensorboard(self):
        print('hello _tensorboard!')

    # def generator(self):
    #     print('hello generator!')
    #
    # def discriminator(self):
    #     print('hello discriminator!')

    def train_step(self):
        fake_y_val, fake_x_val, x_val, y_val = self.sess.run([self.fake_y_imgs, self.fake_x_imgs,
                                                              self.x_imgs, self.y_imgs])


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
            conv6 = tf_utils.padding2d(conv5, p_h=3, p_w=3, pad_type='REFLECT', name='output_padding')
            conv6 = tf_utils.conv2d(conv6, self.image_size[2], k_h=7, k_w=7, d_h=1, d_w=1,
                                     padding='VALID', name='output_conv')
            output = tf_utils.tanh(conv6, name='output_tanh', is_print=True)

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output


class Discriminator(object):
    def __init__(self, name='', ndf=64, norm='instance', _ops=None):
        self.name = name
        self.ndf = ndf
        self.norm = norm
        self._ops = _ops
        self.reuse = False

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H/2, W/2, 64)
            conv1 = tf_utils.conv2d(x, self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv1_conv')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu', is_print=True)

            # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
            conv2 = tf_utils.conv2d(conv1, 2*self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm')
            conv2 = tf_utils.lrelu(conv2, name='conv2_lrelu', is_print=True)

            # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
            conv3 = tf_utils.conv2d(conv2, 4*self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm')
            conv3 = tf_utils.lrelu(conv3, name='conv3_lrelu', is_print=True)

            # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
            conv4 = tf_utils.conv2d(conv3, 8*self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv4_conv')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.lrelu(conv4, name='conv4_lrelu', is_print=True)

            # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
            conv5 = tf_utils.conv2d(conv4, 1, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                                    name='conv5_conv')
            output = tf_utils.sigmoid(conv5, name='output_sigmoid', is_print=True)

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output
