# !/usr/bin/env python

import tensorflow as tf
import numpy as np
import datetime
import time
import os

def main(_):
    t1 = datetime.datetime.now()

    train_X = np.linspace(-1.0, 1.0, 100)
    train_Y = 2.0 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10.0

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    w = tf.Variable(0.0, name="weight")
    b = tf.Variable(0.0, name="bias")
    loss = tf.square(Y - tf.mul(X, w) - b)

    global_step = tf.Variable(0)

    train_op = tf.train.AdagradOptimizer(0.01).minimize(
        loss, global_step=global_step)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step < 1000000:
            for (x, y) in zip(train_X, train_Y):
                _, step = sess.run([train_op, global_step], feed_dict={X: x, Y: y})
            loss_value = sess.run(loss, feed_dict={X: x, Y: y})
            if step % 50000 == 0:
                print time.time()
            print("Step: {}, loss: {}".format(step, loss_value))

    t2 = datetime.datetime.now()
    print("Total Time : {}".format(t2 - t1))


if __name__ == "__main__":
    tf.app.run()