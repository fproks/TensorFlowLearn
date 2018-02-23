# encoding= utf-8
import tensorflow as tf
import lib.input_data_MNIST

x = tf.placeholder(float, [None, 784])
W = tf.Variable(tf.zeros([784.10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.multiply(x, W) + b)
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
y_ = tf.placeholder(float, [None, 10])
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
mnist = lib.input_data_MNIST.read_data_sets("MNIST_data/", one_hot=True)
for i in range(1000):
    batch_xs, batch_ys = mnist
