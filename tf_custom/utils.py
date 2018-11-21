import string
import os
import random
import yaml
import numpy as np
import tensorflow as tf

OPTIMIZERS = {
    'gd': tf.train.GradientDescentOptimizer,
    'adam': tf.train.AdamOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'momentum': tf.train.MomentumOptimizer
}


def tensorboard(fn):
    pass


def get_optimizer(optimizer, learning_rate):
    return OPTIMIZERS[optimizer](learning_rate=learning_rate)


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def random_string(length=6, source=string.ascii_letters+string.digits):
    return ''.join(random.choice(source) for _ in range(length))


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    def _expand_vars(d):
        for k, v in d.items():
            if type(v) == str:
                d[k] = os.path.expandvars(v)
            elif type(v) == dict:
                _expand_vars(v)

    _expand_vars(config)
    return config


def plot_pixels(pixels):
    if isinstance(pixels, list):
        a, b = pixels[0].shape
        x, y = squarish(len(pixels))
        all_pixels = np.empty((a*x, b*y))
        for i in range(x):
            for j in range(y):
                all_pixels[i*a:(i+1)*a, j*b:(j+1)*b] = pixels[i*x + j]
        pixels = all_pixels

    plt.figure()
    plt.imshow(pixels, cmap='gray')
    plt.show()


def squarish(n):
    for i in reversed(range(1, int(n**0.5)+1)):
        if n % i == 0:
            return (i, n // i)


def factors(n):
    s = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            s.update([i, n // i])
    return s
