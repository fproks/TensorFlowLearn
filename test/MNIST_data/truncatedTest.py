# encoding= utf-8
import tensorflow as tf

a = tf.Variable(tf.truncated_normal([5, 5, 2], stddev=0.1))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(a))

a = tf.Variable(tf.random_normal([4, 4, 3], mean=0.0, stddev=1.0, dtype=tf.float32))
b = tf.Variable(tf.constant(0.1, shape=[3]))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(a))
    # print('-----------------------------------------------------')
    # print(sess.run(b))
    # print('====================================================')
    # print(sess.run(a+b))

a = tf.constant([
    [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0],
     [8.0, 7.0, 6.0, 5.0],
     [4.0, 3.0, 2.0, 1.0]],
    [[4.0, 3.0, 2.0, 1.0],
     [8.0, 7.0, 6.0, 5.0],
     [1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0]]
])

a = tf.reshape(a, [1, 4, 4, 2])
pool = tf.nn.max_pool(a, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
with tf.Session() as sess:
    print(sess.run(a))
    print (";;;;;;;;;;;;;;;;;")
    p =sess.run(pool)
    #print(sess.run(pool))
    sess.run(tf.Print(p,[p]))
