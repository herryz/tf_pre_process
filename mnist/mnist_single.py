# import input_data
# import tensorflow as tf
#
#
#
# sess = tf.Session()
# sess.run(init)
# # train the data
# for i in range(10):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print "accuracy", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
# prediction=tf.argmax(y,1)
# print "predictions", prediction.eval(feed_dict={x: mnist.test.images}, session=sess)

# !/usr/bin/env python
import tensorflow as tf
import numpy as np
import datetime


def main(_):
    t1 = datetime.datetime.now()
    mnist = input_data.read_data_sets('data', one_hot=True)
    x = tf.placeholder("float", shape=[None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder("float", shape=[None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_op = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)
    # train data and get results for batches

    saver = tf.train.Saver()
    init = tf.initialize_all_variables()


    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step < 125000:
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, step = sess.run([train_op, global_step], feed_dict={x: batch_xs, y_: batch_ys})
            print("Step %d in task %d" % (step, FLAGS.task_index))

    print("accuracy: %f" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                         y_: mnist.test.labels}))
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
    t2 = datetime.datetime.now()
    print("Total Time : {}".format(t2 - t1))


if __name__ == "__main__":
    tf.app.run()
