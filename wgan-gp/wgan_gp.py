# ---------------------------------------------------------
# TensorFlow WGAN-GP Implementation for Day2Night
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
