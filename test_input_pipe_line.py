"""

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
import tensorflow as tf
import unittest

import input_pipe_line

class test_input_pipe_line(unittest.TestCase):
    def test_imagenet_data_loader(self):

        tfrecord_data_folder = '/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/tfrecord/tfrecord_ImageNet'

        data = input_pipe_line.imagenet_data_loader(dataset_dir=tfrecord_data_folder,
                                                    mode=tf.estimator.ModeKeys.TRAIN,
                                                    preprocess_fn=None,
                                                    batch_size=8)


