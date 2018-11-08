# ---------------------------------------------------------
# Tensorflow VAE Implementation for Day2Night Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import random
import numpy as np
from datetime import datetime

import utils as utils

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


def _init_logger(flags, log_path):
    if flags.is_train:
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        # file handler
        file_handler = logging.FileHandler(os.path.join(log_path, 'dataset.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)


def dataset(flags, dataset_name, image_size, log_path=None):
    if flags.is_train:
        _init_logger(flags, log_path)  # init logger

    if dataset_name == 'paired':
        return Paired(dataset_name, image_size)
    else:
        raise NotImplementedError


class Paired(object):
    def __init__(self, dataset_name, image_size):
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.num_trains, self.num_vals = 0, 0
        self.test_index = 0

        self.train_data_path = '../data/{}/train'.format(self.dataset_name)
        self.val_data_path = '../data/{}/val'.format(self.dataset_name)

        self.is_gray = False
        self._load_data()

    def _load_data(self):
        logger.info('Load {} dataset...'.format(self.dataset_name))

        self.train_data = utils.all_files_under(self.train_data_path, extension='.png')
        self.val_data = utils.all_files_under(self.val_data_path, extension='.png')

        self.num_trains = len(self.train_data)
        self.num_vals = len(self.val_data)

        logger.info('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size=1, which_direction=0):
        random.seed(datetime.now())  # set random seed
        batch_files = np.random.choice(self.train_data, batch_size, replace=False)

        data_x, data_y = [], []
        for batch_file in batch_files:
            batch_x, batch_y = utils.load_data(image_path=batch_file, which_direction=which_direction,
                                               is_gray_scale=self.is_gray, img_size=self.image_size)
            data_x.append(batch_x)
            data_y.append(batch_y)

        batch_ximgs = np.asarray(data_x).astype(np.float32)  # list to array
        batch_yimgs = np.asarray(data_y).astype(np.float32)  # list to array

        return batch_ximgs, batch_yimgs

    def val_next_batch(self, batch_size=1, which_direction=0, is_train=True):
        if is_train:
            random.seed(datetime.now())  # set random seed
            batch_files = np.random.choice(self.val_data, batch_size, replace=False)
        else:
            batch_files = self.val_data[self.test_index:self.test_index + batch_size]
            self.test_index += batch_size

        data_x, data_y = [], []
        for batch_file in batch_files:
            batch_x, batch_y = utils.load_data(image_path=batch_file, flip=False, is_test=True,
                                               which_direction=which_direction, is_gray_scale=self.is_gray,
                                               img_size=self.image_size)

            data_x.append(batch_x)
            data_y.append(batch_y)

        batch_ximg = np.asarray(data_x).astype(np.float32)  # list to array
        batch_yimg = np.asarray(data_y).astype(np.float32)  # list to array

        return batch_ximg, batch_yimg
