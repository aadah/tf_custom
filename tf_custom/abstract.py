import os
import tensorflow as tf
import tf_custom.utils as utils


class Module:
    def __init__(self, *args, **kwargs):
        if 'name' in kwargs:
            self.name = kwargs['name']
            del kwargs['name']
        else:
            self.name = '{}.{}'.format(self.__class__.__name__,
                                       utils.random_string())

        with tf.variable_scope(self.name) as vs:
            self.vs = vs
            self.init(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tf.variable_scope(self.vs):
            return self.call(*args, **kwargs)

    def init(self, *args, **kwargs):
        raise NotImplemented

    def call(self, *args, **kwargs):
        raise NotImplemented

    @property
    def variables(self):
        return tf.trainable_variables(self.vs.name)


class Model:
    def __init__(self, name=None,
                 params=None,
                 training=True,
                 initializer=None,
                 reuse=None):
        if params is not None:
            self.params = params
        else:
            self.params = {}

        if 'name' is not None:
            self.name = name
        else:
            self.name = '{}.{}'.format(self.__class__.__name__,
                                       utils.random_string())

        self.training = training

        with tf.variable_scope(self.name,
                               initializer=initializer,
                               reuse=reuse) as vs:
            self.vs = vs
            self.init()

    def __call__(self, *args, **kwargs):
        with tf.variable_scope(self.vs):
            return self.call(*args, **kwargs)

    def init(self):
        '''
        In this function, you have access to self.params.
        '''

        raise NotImplemented

    def call(self, *args, **kwargs):
        raise NotImplemented

    def build(self):
        raise NotImplemented

    def train(self, dataset):
        raise NotImplemented

    def compile(self):
        self.build()
        self.saver = tf.train.Saver()
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        return self

    def init_vars(self):
        tf.global_variables_initializer().run()
        return self

    def save(self, model_dir):
        self.saver.save(self.sess, os.path.join(model_dir, self.name))
        return self

    def load(self, model_dir):
        self.saver.restore(self.sess, os.path.join(model_dir, self.name))
        return self

    def close(self):
        self.sess.close()
        return self

    @property
    def variables(self):
        return tf.trainable_variables(self.vs.name)
