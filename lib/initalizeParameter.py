# -*- coding: utf-8 -*-
import tensorflow as tf


# 从正态分布中输出随机值。
# http://blog.csdn.net/fireflychh/article/details/73692183
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# http://blog.csdn.net/mao_xiao_feng/article/details/78004522
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# http://blog.csdn.net/mao_xiao_feng/article/details/53453926
'''
max_pool 与conv2d相似,所不同的是,他是去所覆盖区域内的最大值代表这个区域
'''
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
