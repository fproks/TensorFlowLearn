# encoding= utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import lib.input_data

FLAGS = None
mni = lib.input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)  # matmul 矩阵相乘
'''
softmax 回归
 计算每一个像素点的分布权重 以及整副图像对于某个发生概率的偏移量
 该算法中，假定获取了分布权重及偏移量，则某个图像关于 所有概率的计算就是：
 图像 * 权重W +b 得到的是一个没有进行归一化,softmax 的作用就是进行概率的归一化
 现在需要的问题是:得到w 和b 
'''

'''
下面是用了交叉熵去对得到的结果进行评估
y_为真实分布概率
'''
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

'''
tensorFlow 根据你给定的图,使用反向传播算法进行计算
上面我们给定了一系列的节点P
反响传播算法中,头发根据图中的损失函数不断对W 和b 进行迭代,直到corss_entropy最小
梯度下降
'''
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mni.train.next_batch(100)  # 每次选取100组数据
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # 执行训练图
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 获取训练结果与真实标签的结果对比
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 计算正确率  //这是个图
print(sess.run(accuracy, feed_dict={x: mni.test.images, y_: mni.test.labels}))
'''
最后一步的测试里面,
用到了accuracy tensor,
accuracy 使用了correct_prediction tensor
correct_prediction 使用了  y 和y_ 这两个tensor
y 这个tensor的得到方式是y = tf.nn.softmax(tf.matmul(x, W) + b)  # matmul 矩阵相乘
此过程中,W b 这两个图是动态的,可更改的,过程保存的,但是其他的都不是,都是一次计算过程的结果,不保留
因此,在这里使用y的时候,之前的结果已经被擦除了
因此,在计算accuracy 图时,会将x 和y_ 带入,从 y 这里开始计算
'''