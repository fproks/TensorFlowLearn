from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

NUM_CLASSES = 10  # MNIST类别个数
IMAGE_SIZE = 28  # 28*28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE  # 图像大小

HIDDEN1_UNITS = 128  # 第一层的节点数目
HIDDEN2_UNITS = 32  # 第二层的节点数目
LEARNING_RATE = 0.01  # 学习速率
