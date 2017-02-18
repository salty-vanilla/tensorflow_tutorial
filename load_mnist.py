# coding:utf-8
try:
   import cPickle as pickle
except:
   import pickle
import gzip
from collections import namedtuple
from dataset import Dataset
import os


valid_num = 10000
origin = 'https://s3.amazonaws.com/img-datasets/mnist.pkl.gz'


def get_file(url, dst_path):
    import urllib
    try:
        urllib.urlretrieve(url, dst_path)
    except:
        from urllib import request
        request.urlretrieve(url, dst_path)


def load_mnist(data_dir,
               flatten=False, one_hot=True):
    data_path = os.path.join(data_dir, "mnist.pkl.gz")

    if not os.path.exists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        get_file(origin, data_path)

    f = gzip.open(data_path, 'rb')
    data = pickle.load(f, encoding='bytes')
    (x_train, y_train), (x_test, y_test) = data

    # valid
    x_valid = x_train[:valid_num]
    y_valid = y_train[:valid_num]

    # train
    x_train = x_train[valid_num:]
    y_train = y_train[valid_num:]

    if flatten:
        x_train = x_train.reshape(-1, 28 * 28)
        x_valid = x_valid.reshape(-1, 28 * 28)
        x_test = x_test.reshape(-1, 28 * 28)

    datasets = namedtuple('Datasets', ['train', 'validation', 'test'])

    datasets.train = Dataset(images=x_train, labels=y_train, one_hot=one_hot, normalization=True)
    datasets.valid = Dataset(images=x_valid, labels=y_valid, one_hot=one_hot, normalization=True)
    datasets.test = Dataset(images=x_test, labels=y_test, one_hot=one_hot, normalization=True)

    return datasets

if __name__ == "__main__":
    load_mnist(flatten=True)
