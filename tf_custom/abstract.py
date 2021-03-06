import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
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

        self.compile()

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
        if tf.executing_eagerly():
            self.saver = tfe.Saver(self.variables)
        else:
            self.build()
            self.saver = tf.train.Saver(self.variables)
            self.sess = tf.InteractiveSession()
            self.init_vars()
        return self

    def init_vars(self):
        tf.global_variables_initializer().run()
        return self

    def save(self, models_dir, global_step=None):
        path = os.path.join(models_dir, self.name, self.name)
        if tf.executing_eagerly():
            self.saver.save(path, global_step=global_step)
        else:
            self.saver.save(self.sess, path, global_step=global_step)
        return self

    def load(self, models_dir, global_step=None):
        path = os.path.join(models_dir, self.name, self.name)
        path = '{}-{}'.format(path, global_step) if global_step is not None else path
        if tf.executing_eagerly():
            self.saver.restore(path)
        else:
            self.saver.restore(self.sess, path)
        return self

    def close(self):
        self.sess.close()
        return self

    @property
    def variables(self):
        if tf.executing_eagerly():
            return self._trainable_variables(self)
        return tf.trainable_variables(self.vs.name)

    def _trainable_variables(self, obj):
        vars = []
        if isinstance(obj, (tf.Variable, tfe.Variable)):
            vars.append(obj)
        elif isinstance(obj, (Model, Module)):
            for obj2 in obj.__dict__.values():
                vars.extend(self._trainable_variables(obj2))
        elif isinstance(obj, (list, tuple)):
            for obj2 in obj:
                vars.extend(self._trainable_variables(obj2))
        elif isinstance(obj, dict):
            for obj_key, obj_val in obj.items():
                vars.extend(self._trainable_variables(obj_key))
                vars.extend(self._trainable_variables(obj_val))
        return vars
