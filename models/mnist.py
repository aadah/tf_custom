import argparse
import numpy as np
import tensorflow as tf
import tf_custom.utils as utils

from tf_custom.modules import SingleLayer
from tf_custom.abstract import Model
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score


OPTIM = {
    'gd': tf.train.GradientDescentOptimizer
}


class NN(Model):
    def init(self):
        self.layers = [SingleLayer(self.params['data_dim'], self.params['hidden_dim'],
                                   name='layer1',
                                   use_bias=True,
                                   activation=self.params['non_lin'])]

        for i in range(self.params['num_hidden_layers']):
            layer = SingleLayer(self.params['hidden_dim'], self.params['hidden_dim'],
                                name='layer{}'.format(i+2),
                                use_bias=True,
                                activation=self.params['non_lin'])
            self.layers.append(layer)

        self.layers.append(SingleLayer(self.params['hidden_dim'], 10,
                                       name='layer{}'.format(len(self.layers)+1),
                                       use_bias=True,
                                       activation=self.params['non_lin']))

    def build(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.params['data_dim']])
        self.Y = tf.placeholder(tf.float32, shape=[None, 10])
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.Y_ = self(self.X)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,
                                                                           logits=self.Y_))

        optimizer = OPTIM['gd']
        self.train_op = optimizer(self.lr).minimize(self.loss)

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def train(self, dataset):
        for i in range(self.params['train']['num_iter']):
            X, Y = dataset.train.next_batch(self.params['train']['batch_size'])
            lr = 1.0 / np.power(i + 2, self.params['train']['gamma'])
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.X: X,
                self.Y: Y,
                self.lr: lr
            })

            if (i+1) % 1000 == 0:
                print('Iter {}:\t{}\t(lr: {})'.format(i+1, loss, lr))

    def test(self, dataset):
        X, Y = dataset.test.images, dataset.test.labels
        Y_ = self.Y_.eval(feed_dict={self.X: X})

        y = np.argmax(Y, axis=1)
        y_ = np.argmax(Y_, axis=1)

        print(accuracy_score(y, y_))

def main(args):
    opts = utils.load_options(args.options)
    mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)
    model = NN(name=opts['name'],
               initializer=tf.contrib.layers.xavier_initializer(),
               params=opts['params']).compile()

    if args.test:
        model.load('../data/models/')
        model.test(mnist)
    else:
        model.init_vars()
        model.train(mnist)
        model.save('../data/models/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--options')
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()

    main(args)
