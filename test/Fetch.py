# encoding= utf-8
import tensorflow as tf

input1 = tf.placeholder(tf.float32)  # placeholder占位符
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:   # feed_dict 临时数据
    print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))
