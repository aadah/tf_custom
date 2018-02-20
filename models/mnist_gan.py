import argparse
import tensorflow as tf
import tf_custom.utils as utils

from tensorflow.examples.tutorials.mnist import input_data
from tf_custom.modules import SingleLayer, UniformSampler
from tf_custom.abstract import Model


OPTIM = {
    'adam': tf.train.AdamOptimizer,
    'gd': tf.train.GradientDescentOptimizer
}


class Generator(Model):
    def init(self):
        self.sampler = UniformSampler([self.params['sample_low'] for _ in range(self.params['sample_dim'])],
                                      [self.params['sample_high'] for _ in range(self.params['sample_dim'])])

        self.layers = [SingleLayer(self.params['sample_dim'], self.params['gen_hidden_dim'],
                                   name='layer1',
                                   use_bias=True,
                                   activation=self.params['gen_non_lin'])]
        for i in range(self.params['gen_num_hidden_layers']):
            layer = SingleLayer(self.params['gen_hidden_dim'], self.params['gen_hidden_dim'],
                                name='layer{}'.format(i+2),
                                use_bias=True,
                                activation=self.params['gen_non_lin'])
            self.layers.append(layer)

        self.layers.append(SingleLayer(self.params['gen_hidden_dim'], self.params['data_dim'],
                                       name='layer{}'.format(len(self.layers)+1),
                                       use_bias=True,
                                       activation='sigmoid'))

    def call(self, sample_size):
        inputs = self.sampler(sample_size)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class Discriminator(Model):
    def init(self):
        self.layers = [SingleLayer(self.params['data_dim'], self.params['dis_hidden_dim'],
                                   name='layer1',
                                   use_bias=True,
                                   dropout_prob=self.params['dropout_prob'],
                                   activation=self.params['dis_non_lin'])]
        for i in range(self.params['dis_num_hidden_layers']):
            layer = SingleLayer(self.params['dis_hidden_dim'], self.params['dis_hidden_dim'],
                                name='layer{}'.format(i+2),
                                use_bias=True,
                                dropout_prob=self.params['dropout_prob'],
                                activation=self.params['dis_non_lin'])
            self.layers.append(layer)

        self.layers.append(SingleLayer(self.params['dis_hidden_dim'], 2,
                                       name='layer{}'.format(len(self.layers)+1),
                                       use_bias=True,
                                       activation=self.params['dis_non_lin']))

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class GAN(Model):
    def init(self):
        self.generator = Generator(name='Generator',
                                   params=self.params)
        self.discriminator = Discriminator(name='Discriminator',
                                           params=self.params)

    def call(self, sample_size, real_data):
        fake_data = self.generator(sample_size)

        fake_logits = self.discriminator(fake_data)
        real_logits = self.discriminator(real_data)

        return fake_logits, real_logits

    def build(self):
        self.sample_size = tf.placeholder(tf.int32, shape=[])
        self.real_data = tf.placeholder(tf.float32, shape=[None, self.params['data_dim']])

        fake_logits, real_logits = self(self.sample_size, self.real_data)
        fake_log_probs = tf.nn.log_softmax(fake_logits)[:,0]
        real_log_probs = tf.nn.log_softmax(real_logits)[:,1]

        self.gen_loss = tf.reduce_mean(fake_log_probs)
        self.dis_loss = - (tf.reduce_mean(real_log_probs) + self.gen_loss)

        optimizer = OPTIM[self.params['train']['optimizer']]
        lr = self.params['train']['lr']

        self.gen_train_op = optimizer(lr).minimize(self.gen_loss, var_list=self.generator.variables)
        self.dis_train_op = optimizer(lr).minimize(self.dis_loss, var_list=self.discriminator.variables)

    def train(self, dataset):
        sample_size = self.params['train']['sample_size']
        for i in range(self.params['train']['num_iter']):
            for j in range(self.params['train']['k']):
                real_data, _ = dataset.next_batch(self.params['train']['batch_size'])
                dis_loss, _ = self.sess.run([self.dis_loss, self.dis_train_op],
                                            feed_dict={
                                                self.sample_size: sample_size,
                                                self.real_data: real_data
                                            })

            gen_loss, _ = self.sess.run([self.gen_loss, self.gen_train_op],
                                        feed_dict={
                                            self.sample_size: sample_size
                                        })

            if (i+1) % 1000 == 0:
                print('Iter {}:\tDIS: {}\tGEN: {}'.format(i+1, dis_loss, gen_loss))


def main(args):
    opts = utils.load_options(args.options)
    mnist = input_data.read_data_sets("../data/MNIST_data/")

    gan = GAN(name=opts['name'],
              initializer=tf.contrib.layers.xavier_initializer(),
              params=opts['params']).compile()

    if args.test:
        gan.load('../data/models/')
        utils.plot_pixels(list(gan.generator(100).eval().reshape(-1,28,28)))
    else:
        gan.init_vars()
        gan.train(mnist.train)
        gan.save('../data/models/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--options')
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()

    main(args)
