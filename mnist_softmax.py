from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import sys
import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data
from load_mnist import load_mnist
import sys


FLAGS = None

batch_size = 100
nb_epochs = 10


def main():
    # Arg parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./input_data/mnist.pkl.gz',
                        help='Type mnist data path')

    args = parser.parse_args()

    data_path = args.data_path

    # Import data
    mnist = load_mnist(flatten=True, data_path=data_path)

    # Create the model
    inputs = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    outputs = tf.matmul(inputs, W) + b

    # Label placeholder
    labels = tf.placeholder(tf.float32, [None, 10])

    # Correct
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Loss and Optimezer
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outputs))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train
    for epoch in range(nb_epochs):
        sys.stdout.write("[epoch] : %d\n" % epoch)
        for iter, (batch_xs, batch_ys) in enumerate(mnist.train.next_batch(batch_size)):
            feed_dict = {inputs: batch_xs, labels: batch_ys}
            _, train_loss, train_accuracy = sess.run([train_step, cross_entropy, accuracy],
                                                     feed_dict=feed_dict)
            sys.stdout.flush()
            sys.stdout.write("\r%d / %d" % ((iter * batch_size), mnist.train.num_data))
            sys.stdout.write("  [train loss] : %f" % train_loss)
            sys.stdout.write("  [train accuracy] %f" % train_accuracy)

        feed_dict = {inputs: mnist.valid.images, labels: mnist.valid.labels}
        valid_loss, valid_accuracy = sess.run([cross_entropy, accuracy],
                                              feed_dict=feed_dict)

        sys.stdout.write("\n[valid loss] : %f" % valid_loss)
        sys.stdout.write("  [valid accuracy] %f\n\n" % valid_accuracy)

    sys.stdout.write("Complete training !!\n")

    # Test trained model
    feed_dict = {inputs: mnist.train.images, labels: mnist.train.labels}
    test_loss, test_accuracy = sess.run([cross_entropy, accuracy],
                                          feed_dict=feed_dict)

    sys.stdout.write("[test loss] : %f" % test_loss)
    sys.stdout.write("  [test accuracy] %f\n" % test_accuracy)


if __name__ == '__main__':
    main()