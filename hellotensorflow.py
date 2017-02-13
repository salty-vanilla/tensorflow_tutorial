# -*- coding: utf-8 -*-
# 参考サイト：http://kivantium.hateblo.jp/entry/2015/11/18/233834


import tensorflow as tf

# String形の定数helloの式を定義
hello = tf.constant('Hello, TensorFlow!')

# Sessionを開始
sess = tf.Session()

# Sessionを実行して表示
print(sess.run(hello))

a = tf.constant(765)
b = tf.constant(346)
print(sess.run(a + b))