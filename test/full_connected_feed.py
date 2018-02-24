import tensorflow as tf
import lib.input_data as input_data
from test import mnist
import math
import os
import time
import sys

batch_size = 100

log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                       '/logs/fully_connected_feed')


def do_eval(sess,  # 训练过的session
            eval_correct,  # The Tensor that returns the number of correct predictions.
            images_placeholder,
            labels_placeholder,
            data_set):  # 待测试样本
    # 测试
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // batch_size  # 求摸
    num_examples = steps_per_epoch * batch_size  # 获取测试数量
    for step in range(steps_per_epoch):
        batch = data_set.next_batch(batch_size=batch_size, fake_data=False)
        true_count += sess.run(eval_correct, feed_dict={images_placeholder: batch[0], labels_placeholder: batch[1]})
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    data_sets = input_data.read_data_sets('MNIST_data', fake_data=False)  # 下载数据
    # 创建训练图像shape 和标记shape
    with tf.Graph().as_default():
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

        print('images_placeholder 的维度为：', images_placeholder.shape)
        print('lables_placeholder 的维度为:', labels_placeholder.shape)

        # 第一层
        with tf.name_scope('hidden1'):
            weight = tf.Variable(
                tf.truncated_normal([mnist.IMAGE_PIXELS, mnist.HIDDEN1_UNITS],
                                    stddev=1.0 / math.sqrt(float(mnist.IMAGE_PIXELS))),
                name='weights')
            biases = tf.Variable(tf.zeros([mnist.HIDDEN1_UNITS]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(images_placeholder, weight) + biases)  # 第一层的计算方法

            print('hidden1 的维度为:', hidden1.shape)

        # 第二层
        with tf.name_scope('hidden2'):
            weight = tf.Variable(
                tf.truncated_normal([mnist.HIDDEN1_UNITS, mnist.HIDDEN2_UNITS],
                                    stddev=1.0 / math.sqrt(float(mnist.HIDDEN1_UNITS))),
                name='weight')
            biases = tf.Variable(tf.zeros([mnist.HIDDEN2_UNITS]), name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weight) + biases)
            print('hidden2 的维度为:', hidden2.shape)

        # 连接层
        with tf.name_scope('softmax_linear'):
            weight = tf.Variable(
                tf.truncated_normal([mnist.HIDDEN2_UNITS, mnist.NUM_CLASSES],
                                    stddev=1.0 / math.sqrt(float(mnist.HIDDEN2_UNITS))),
                name='weight')
            biases = tf.Variable(tf.zeros([mnist.NUM_CLASSES]), name='biases')
            logits = tf.matmul(hidden2, weight) + biases
            print('logits 的维度为:', logits.shape)

        labels = tf.to_int64(labels_placeholder)
        print('labels 维度:', labels.shape)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        print('loss 的维度为:', loss.shape)

        tf.summary.scalar('loss', loss)
        optimizer = tf.train.GradientDescentOptimizer(mnist.LEARNING_RATE)  # 设置下降参数
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)  # 训练目标

        correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
        eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(init)

        for step in range(2000):
            start_time = time.time()
            batch = data_sets.train.next_batch(batch_size, fake_data=False)
            _, loss_value = sess.run([train_op, loss], feed_dict={images_placeholder: batch[0],
                                                                  labels_placeholder: batch[1]})

            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict={images_placeholder: batch[0],
                                                           labels_placeholder: batch[1]})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == 2000:
                checkout_file = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkout_file, global_step=step)
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test)


def main():
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    run_training()


if __name__ == '__main__':
    # tf.app.run(main=main, argv=[sys.argv[0]])
    main()
