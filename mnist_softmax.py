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
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()

    data_path = args.data_path
    verbose = args.verbose

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
        sys.stdout.write("Epoch %d/%d\n" % (epoch, nb_epochs))
        for iter, (batch_xs, batch_ys) in enumerate(mnist.train.next_batch(batch_size)):
            feed_dict = {inputs: batch_xs, labels: batch_ys}
            _, train_loss, train_accuracy = sess.run([train_step, cross_entropy, accuracy],
                                                     feed_dict=feed_dict)

            if verbose == 1:
                length = 30
                percentage = float(iter * batch_size / mnist.train.num_data)
                bar = "[" + "=" * int(length * percentage) + "-" * (length - int(length * percentage)) + "]"
                display = "\r{} / {} {} "\
                          "loss: {:.4f} - acc: {:.4f}"\
                    .format(iter * batch_size, mnist.train.num_data, bar, train_loss, train_accuracy)
                sys.stdout.write(display)
                sys.stdout.flush()

        feed_dict = {inputs: mnist.valid.images, labels: mnist.valid.labels}
        valid_loss, valid_accuracy = sess.run([cross_entropy, accuracy],
                                              feed_dict=feed_dict)
        if verbose == 1:
            display = " - val_loss : {:.4f} - val_acc : {:.4f}\n"\
                .format(valid_loss, valid_accuracy)
            sys.stdout.write(display)

    sys.stdout.write("\nComplete training !!\n")

    # Test trained model
    feed_dict = {inputs: mnist.train.images, labels: mnist.train.labels}
    test_loss, test_accuracy = sess.run([cross_entropy, accuracy],
                                        feed_dict=feed_dict)

    display = "test_loss : {:.4f} - test_acc : {:.4f}\n" \
        .format(test_loss, test_accuracy)
    sys.stdout.write(display)


if __name__ == '__main__':
    main()