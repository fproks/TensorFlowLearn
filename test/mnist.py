# encoding= utf-8
import tensorflow as tf

x = tf.placeholder(float, [None, 784])
W = tf.Variable(tf.zeros([784.10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.multiply(x, W) + b)
'''
softmax 回归
 计算每一个像素点的分布权重 以及整副图像对于某个发生概率的偏移量
'''
