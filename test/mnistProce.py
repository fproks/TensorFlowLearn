# encoding= utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import tensorflow as tf
import lib.input_data
import lib.initalizeParameter as init

FLAGS = None
mni = lib.input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])  # 仅仅是一个占位符, 是一个二阶张量
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # 仅仅是一个占位符

'''
x 的数据应该是 若干张 图像像素个数为784的图片的集合,他是一个二阶张量,一个数据只需要两个坐标来确定
彩色图像是3阶的,因为他还需要一个RGB 选择维度
'''
'''
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))
'''
# y = tf.nn.softmax(tf.matmul(x, w) + b)  # 目标函数

w_conv1 = init.weight_variable([5, 5, 1, 32])  # 随机得到
b_conv1 = init.bias_variable([32])  # 随机得到
x_image = tf.reshape(x, [-1, 28, 28, 1])
'''
http://blog.csdn.net/mao_xiao_feng/article/details/78004522
w_conv1 是一个包含32个卷积 ,每个卷积 是一个5*5*1 的矩阵
一般卷积的通道数和图像的通道数保持一致
w_conv1 是deeplearning  神经网络的第一层的参数,也是需要不断迭代的参数
b_conv1 也是第一层的参数
b_conv1 是一个偏执量,初次赋值为0.1
这个方法都是反向迭代的,因此会不停的纠正
w_conv1 和b_conv1 在开始的时候给定的是随机变量,
而且 w_conv1 是一个四维张量 ,他的第一维有5列,第二维有5列,第三维只有一列,第四维有32个参数
RGB图是一个三维张量,第一维是横坐标,第二维是纵坐标,第三维是RGB 参数
b_conv1 是一个一维张量,这意味这他只有一个参数就可以确定一个数据,这个数据是一个32位的数组
x_image 是将 图像数据变成了一个四维张量, 
reshape 使用的这个-1 是用来推测该维度的大小的,意思是我不知道,让cpu自动推测
原本一副图像的数据是784个,更改张量后数据变成了[1,28,28,1]形式的数据
但是一次不是输入一副数据,是输入n副图像,那么tensor 就变成了[n,28,28,1] 
与下面的conv2d 对应 
'''

# 将数据列变成了图像,其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
'''
tf.nn.conv2d(input, w, strides, padding)
其中 input 为输入，格式为[batch, height, width, channels], 分别为【输入的批次数量、图像的高（行数）、宽（列数）、通道（彩色为3，灰色为1）】
w 为卷积矩阵，二维、分别为[高，宽】
conv2d 获得了的shape 应该为[n,28,28,32] 
即,每张图获得的卷积结果是一个28*28 有32个通道的张量
28 也就是一个卷积在一个方向滑动28步,当不足5*5时补0
卷积出来的结果与偏执量进行相加
relu 是为了保持数据的稀疏性,实践证明,大部分数据是无用的,需要置为0
与sigmoid 为同一个意思,也就是数据的归一化,
这里没有归一化,只是将小于0的数据置为0:g(x) = max(0, x)
'''
# h_conv1[n,28,28,32]
# h_pool1[n,14,14,32]
h_conv1 = tf.nn.relu(init.conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = init.max_pool_2x2(h_conv1)
print(h_conv1.shape)
print(h_pool1.shape)
# w_conv2 [5,5,32,64]  64个卷积的集合,每个卷积32层,每一层5*5
# b_conv2[64]
w_conv2 = init.weight_variable([5, 5, 32, 64])
b_conv2 = init.bias_variable([64])

# h_conv2 [n,14,14,64]
# h_pool2 [n,7,7,64]
h_conv2 = tf.nn.relu(init.conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = init.max_pool_2x2(h_conv2)

print(h_conv2.shape)
print(h_pool2.shape)

# w_fc1 [7*7.64,1024]
w_fc1 = init.weight_variable([7 * 7 * 64, 1024])
b_fc1 = init.bias_variable([1024])

# h_pool2_flat[n,7*7*1024]
# h_fc1[n,1024]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)  # 两层softmax,第一层稀疏,对得到的进行稀疏化

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 设置一个概率使神经元不启用,即不变

w_fc2 = init.weight_variable([1024, 10])
b_fc2 = init.bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)  # 第二层softmax, 对结果进行概率化

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # 损失函数  损失函数越低,拟合度越高
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # Adam 优化器
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mni.train.next_batch(500)
    if i % 50 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d,training accuracy %g" % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print(sess.run(accuracy, feed_dict={x: mni.test.images, y_: mni.test.labels}))
