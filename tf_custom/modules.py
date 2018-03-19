import tensorflow as tf
import tf_custom.abstract as abstract


class SingleLayer(abstract.Module):
    '''
    Multiplies the input through a standard neural network layer.
    '''

    def init(self, input_dim, units,
             use_bias=True,
             activation=None,
             leaky_relu_alpha=None,
             dropout_prob=None,
             weights_initializer=None,
             bias_initializer=None):
        '''
        args:
        input_dim - the dimension of the inputs
        units - number of neurons

        kwargs:
        use_bias - if True, add new bias vector param to dense output
        activation - optional activation module. no activation if None
        '''

        self.input_dim = input_dim
        self.units = units

        self.use_bias = use_bias

        self.activation = SingleLayer.make_activation(activation, leaky_relu_alpha) if activation is not None else lambda x: x
        self.dropout = Dropout(dropout_prob) if dropout_prob is not None else lambda x: x

        self.weights = tf.get_variable('weights',
                                       dtype=tf.float32,
                                       shape=[self.input_dim, self.units],
                                       initializer=weights_initializer)

        if self.use_bias:
            self.bias = tf.get_variable('bias',
                                        dtype=tf.float32,
                                        shape=[self.units],
                                        initializer=bias_initializer)

    @staticmethod
    def make_activation(activation, leaky_relu_alpha):
        if isinstance(activation, Activation):
            return activation

        ACTIVATIONS = {
            'sigmoid': SigmoidActivation,
            'elu': ExpoLUActivation,
            'relu': ReLUActivation,
            'leaky_relu': LeakyReLUActivation,
            'adaptive_leaky_relu': AdaptiveLeakyReLUActivation
        }

        act = ACTIVATIONS[activation]
        if activation in ['leaky_relu', 'adaptive_leaky_relu']:
            return act(leaky_relu_alpha)
        else:
            return act()

    def call(self, inputs):
        '''
        Returns a tensor with dimension dims(inputs)[:-1] + [units]

        args:
        inputs - a tensor of rank >= 2
        '''

        inputs_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, [-1, inputs_shape[-1]])

        outputs = tf.matmul(inputs, self.weights)

        if self.use_bias:
            outputs += self.bias

        outputs = self.activation(outputs)
        outputs_shape = tf.concat([inputs_shape[:-1], [self.units]], 0)
        outputs = tf.reshape(outputs, outputs_shape)
        outputs = self.dropout(outputs)

        return outputs


class FeedForward(abstract.Module):
    def init(self, input_dim, output_dim,
             num_layers=None,
             hidden_dim=None,
             hidden_dims=None,
             use_bias=True,
             activation='relu',
             output_activation=None,
             leaky_relu_alpha=None,
             dropout_prob=None,
             weights_initializer=None,
             bias_initializer=None):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dims = []
        if num_layers is not None:
            if hidden_dim is not None:
                for _ in range(num_layers-1):
                    self.dims.append(hidden_dim)
            else:
                raise Exception("Need to specify `hidden_dim` if using `num_layers`.")
        else:
            if hidden_dims is not None:
                self.dims.extend(hidden_dims)
            else:
                raise Exception("Need to specify `hidden_dims` if not using `num_layers`.")

        self.dims.append(self.output_dim)
        self.num_layers = len(self.dims)

        self.layers = []
        for i, out_dim in enumerate(self.dims):
            in_dim = self.input_dim if i == 0 else self.dims[i-1]
            layer_activation = output_activation if i == self.num_layers - 1 else activation

            layer = SingleLayer(in_dim, out_dim,
                                name='layer{}'.format(i+1),
                                use_bias=use_bias,
                                activation=layer_activation,
                                leaky_relu_alpha=leaky_relu_alpha,
                                dropout_prob=dropout_prob,
                                weights_initializer=weights_initializer,
                                bias_initializer=bias_initializer)

            self.layers.append(layer)

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return tf.reshape(inputs, [-1])


class ChanneledSingleLayer(abstract.Module):
    '''
    Like SingleLayer, but with separate parameters per channel.
    '''

    def init(self, input_dim, units, num_channels,
             use_bias=True,
             activation=None,
             leaky_relu_alpha=None,
             dropout_prob=None,
             weights_initializer=None,
             bias_initializer=None):
        '''
        args:
        input_dim - the dimension of the inputs
        units - number of neurons

        kwargs:
        use_bias - if True, add new bias vector param to dense output
        activation - optional activation module. no activation if None
        '''

        self.input_dim = input_dim
        self.units = units
        self.num_channels = num_channels

        self.use_bias = use_bias

        self.activation = SingleLayer.make_activation(activation, leaky_relu_alpha) if activation is not None else lambda x: x
        self.dropout = Dropout(dropout_prob) if dropout_prob is not None else lambda x: x

        self.weights = tf.get_variable('weights',
                                       dtype=tf.float32,
                                       shape=[self.num_channels, self.input_dim, self.units],
                                       initializer=weights_initializer)

        if self.use_bias:
            self.bias = tf.get_variable('bias',
                                        dtype=tf.float32,
                                        shape=[self.num_channels, self.units],
                                        initializer=bias_initializer)

    @staticmethod
    def make_activation(activation, leaky_relu_alpha):
        if isinstance(activation, Activation):
            return activation

        ACTIVATIONS = {
            'sigmoid': SigmoidActivation,
            'elu': ExpoLUActivation,
            'relu': ReLUActivation,
            'leaky_relu': LeakyReLUActivation,
            'adaptive_leaky_relu': AdaptiveLeakyReLUActivation
        }

        act = ACTIVATIONS[activation]
        if activation in ['leaky_relu', 'adaptive_leaky_relu']:
            return act(leaky_relu_alpha)
        else:
            return act()

    def call(self, inputs, channels):
        '''
        Returns a tensor with dimension dims(inputs)[:-1] + [units]

        args:
        inputs - a tensor of rank >= 2
        channels - channels to use per input
        '''

        inputs_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, [-1, inputs_shape[-1]])
        inputs = tf.expand_dims(inputs, axis=1)

        weights = tf.nn.embedding_lookup(self.weights, channels)

        outputs = tf.matmul(inputs, weights)
        outputs = tf.squeeze(outputs, axis=1)

        if self.use_bias:
            bias = tf.nn.embedding_lookup(self.bias, channels)
            outputs += bias

        outputs = self.activation(outputs)
        outputs_shape = tf.concat([inputs_shape[:-1], [self.units]], 0)
        outputs = tf.reshape(outputs, outputs_shape)
        outputs = self.dropout(outputs)

        return outputs


class Conv2DLayer(abstract.Module):
    pass


class Dropout(abstract.Module):
    def init(self, keep_prob, training=True):
        self.keep_prob = keep_prob
        self.training = training

    def call(self, inputs):
        return tf.layers.dropout(inputs,
                                 rate=self.keep_prob,
                                 training=self.training)


class Activation(abstract.Module):
    '''
    Custom activation module.
    '''

    def init(self, fn):

        '''
        args:
        fn - any element-wise tensorflow function

        kwargs:
        name - unique string for this variable. auto-generated if not provided
        '''

        self.fn = fn

    def call(self, inputs):
        return self.fn(inputs)


class SigmoidActivation(Activation):
    def init(self):
        super().init(tf.nn.sigmoid)


class ExpoLUActivation(Activation):
    def init(self):
        super().init(tf.nn.elu)


class ReLUActivation(Activation):
    def init(self):
        super().init(tf.nn.relu)


class LeakyReLUActivation(Activation):
    def init(self, alpha=0.2):
        self.alpha = alpha if alpha is not None else 0.2
        super().init(lambda x: tf.nn.leaky_relu(x, alpha=self.alpha))


class AdaptiveLeakyReLUActivation(LeakyReLUActivation):
    def init(self, initial_alpha=None):
        self.initial_alpha = initial_alpha if initial_alpha is not None else 0.2
        alpha = tf.Variable(self.initial_alpha,
                            name='alpha',
                            dtype=tf.float32,
                            trainable=True)
        super().init(alpha=alpha)


class BatchNormalization(abstract.Module):
    def init(self, inputs_shape,
             epsilon=1e-3):

        if not isinstance(inputs_shape, list):
            inputs_shape = [inputs_shape]

        self.inputs_shape = inputs_shape
        self.epsilon = epsilon

        self.scale = tf.get_variable('scale',
                                     dtype=tf.float32,
                                     shape=self.inputs_shape,
                                     initializer=tf.ones_initializer())
        self.offset = tf.get_variable('offset',
                                      dtype=tf.float32,
                                      shape=self.inputs_shape,
                                      initializer=tf.zeros_initializer())

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[0],
                                       keep_dims=False)

        outputs = tf.nn.batch_normalization(inputs, mean, variance,
                                            self.offset,
                                            self.scale,
                                            self.epsilon)

        return outputs


class Sampler(abstract.Module):
    def init(self, distribution):
        self.distribution = distribution

    def call(self, num_samples):
        return self.distribution.sample(num_samples)


class UniformSampler(Sampler):
    def init(self, low, high):
        super().init(tf.contrib.distributions.Uniform(low=low, high=high))


class GaussianSampler(Sampler):
    def init(self, loc, scale):
        super().init(tf.contrib.distributions.Normal(loc=loc, scale=scale))
