import numpy as np
from collections import Counter
import random

class Dataset(object):

    def __init__(self, images, labels, one_hot=True, normalization=False):
        self.images = images.astype('float32')
        self.labels = labels.astype('int32')
        self.num_data = len(self.images)

        if normalization:
            self.images /= 255

        if one_hot:
            self.labels = self.convert_to_one_hot(self.labels)

        assert len(self.images) == len(self.labels)

    @staticmethod
    def convert_to_one_hot(labels):
        counter = Counter(labels)
        label_num = len(list(counter.keys()))

        one_hot = np.zeros((labels.shape[0], label_num))

        for label, oh in zip(labels, one_hot):
            oh.put(label, 1)

        return one_hot

    def next_batch(self, batch_size, shuffle=True):
        indexes = [i for i in range(0, self.num_data)]
        max_iter = int(self.num_data / batch_size)

        if shuffle:
            random.shuffle(indexes)

        for iter in range(max_iter):
            batch_indexes = indexes[batch_size * iter: batch_size * (iter + 1)]
            yield self.images[batch_indexes], self.labels[batch_indexes]

        if not self.num_data == batch_size * (iter + 1):
            batch_indexes = indexes[batch_size * (iter + 1):]
            yield self.images[batch_indexes], self.labels[batch_indexes]


