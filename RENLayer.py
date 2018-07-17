# First we will import the abstract class 'Layer' which every custom layer's class should implement
# Now since our model has 'trainable' parameters, we need to import module which deals with initialization of them
# import the keras backend module which deals with backend in a rather abstract manner
# Import other std. modules also which will deal with optimization, adding Fully Conntected layers for classification etc.
from keras import backend as K
import tensorflow as tf
from keras.layers import Recurrent
from keras.engine.topology import Layer
import functools
from keras import activations, initializers


def prelu_func(features, initializer=None, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU', initializer=initializer):
        alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg
prelu = functools.partial(prelu_func, initializer=tf.constant_initializer(1.0))

# Define a class which will wrap all the model details
# It should inherit the abstract parent class 'Layer' which is the parent class for all layers in Keras


class RENL(Layer):
    '''Output layer.  Extends layer.  Equation 6 from paper.'''

    # Initialise the parameters
    def __init__(self, embedding_size, vocab_size, num_blocks, activation, **kwargs):
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.activation = activation
        # if activation == 'prelu':
        #     self.activation = prelu
        # else:
        #     self.activation = activations.get(activation)
        self.supports_masking = True

        super(RENL, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialise trainable weights
        self.R = self.add_weight((self.embedding_size, self.vocab_size), initializer='normal', name='R', trainable=True)
        self.H = self.add_weight((self.embedding_size, self.embedding_size), initializer='normal', name='H', trainable=True)

        self.supports_masking = True
        super(RENL, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return None, self.vocab_size

    def call(self, x):
        """
            Implementation of Section 2.3, Equation 6. This module is also described in more detail here:
            [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852).
        """
        last_state = x[0]
        self.encoded_query = x[1]
        last_state = tf.stack(tf.split(last_state, self.num_blocks, axis=1), axis=1)
        _, _, embedding_size = last_state.get_shape().as_list()

        # Use the encoded_query to attend over memories
        # (hidden states of dynamic last_state cell blocks)
        attention = tf.reduce_sum(last_state * self.encoded_query, axis=2)

        # Subtract max for numerical stability (softmax is shift invariant)
        attention_max = tf.reduce_max(attention, axis=-1, keep_dims=True)
        attention = tf.nn.softmax(attention - attention_max)
        attention = tf.expand_dims(attention, axis=2)

        # Weight memories by attention vectors
        u = tf.reduce_sum(last_state * attention, axis=1)

        # R acts as the decoder matrix to convert from internal state to the output vocabulary size

        # q = tf.reduce_sum(self.encoded_query, axis=1)
        q = tf.squeeze(self.encoded_query, axis=1)
        y = tf.matmul(self.activation(q + tf.matmul(u, self.H)), self.R)
        return y


class REN(Recurrent):
    '''Actual dynamic memory cell for Recurrent Entity Network.  Extends Recurrent.  Equations 2-5 from paper.'''
    # Initialise the parameters
    def __init__(self,
                 initial_batch_size,
                 units,
                 num_blocks,
                 num_units_per_block,
                 vocab_size,
                 keys,
                 activation,
                 weights=None,
                 initializer='normal',
                 bias_initializer='zeros',
                 use_bias=True,
                 **kwargs):
        super(REN, self).__init__(**kwargs)
        self.units = units
        self._num_blocks = num_blocks
        self._num_units_per_block = num_units_per_block
        self._vocab_size = vocab_size
        self._keys = keys
        self._activation = activation
        # self._activation = activation
        # if activation == 'prelu':
        #     self._activation = prelu
        # else:
        #     self._activation = activations.get(activation)
        self._initializer = initializers.random_normal(stddev=0.1)
        # self.ortho_initializer = tf.orthogonal_initializer(gain=1.0)
        self.initial_batch_size = initial_batch_size
        self.bias_initializer = initializers.get(bias_initializer)
        self.supports_masking = True
        self.use_bias = use_bias
        self.initial_weights = weights

    @property
    def output_size(self):
        """ return the total output size of the cell across all blocks """

        return self._num_blocks * self._num_units_per_block

    @property
    def state_size(self):
        """ return the total state size of the cell across all blocks """

        return self._num_blocks * self._num_units_per_block

    def zero_state(self, batch_size):

        """ initialize the memory to the key values """
        zero_state = tf.concat([tf.expand_dims(key, axis=0) for key in self._keys], axis=1)
        zero_state_batch = tf.tile(zero_state, [batch_size, 1])
        return [zero_state_batch]

    def get_gate(self, state_j, key_j, inputs):
        """
        Implements the gate (scalar for each block). Equation 2:

        g_j <- \sigma(s_t^T h_j + s_t^T w_j)

        """
        a = tf.reduce_sum(inputs * state_j, axis=1)
        b = tf.reduce_sum(inputs * key_j, axis=1)
        return K.sigmoid(a + b)

    def get_candidate(self, state_j, key_j, inputs, U, V, W, U_bias):

        """
        Represents the new memory candidate that will be weighted by the gate value
        and combined with the existing memory. Equation 3:

        h_j^~ <- \phi(U h_j + V w_j + W s_t)
        """
        key_V = tf.matmul(key_j, V)
        state_U = tf.matmul(state_j, U) + U_bias
        inputs_W = tf.matmul(inputs, W)
        return self._activation(state_U + key_V + inputs_W)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        self.states = [None]
        if self.stateful:
            self.reset_states()

        self.U = self.add_weight((self._num_units_per_block, self._num_units_per_block), initializer=self._initializer, name='U',trainable=True)
        self.V = self.add_weight((self._num_units_per_block, self._num_units_per_block), initializer=self._initializer, name='V',trainable=True)
        self.W = self.add_weight((self._num_units_per_block, self._num_units_per_block), initializer=self._initializer, name='W',trainable=True)
        self.U_bias = self.add_weight((self._num_units_per_block,), initializer=self._initializer, name='U_bias',trainable=True)

        # Build te activation layer
        self._activation.build((self.initial_batch_size, self._num_units_per_block))

        # Add activation trainable weights to model
        self.trainable_weights += self._activation.trainable_weights
        super(REN, self).build(input_shape)

    def preprocess_input(self, inputs, training=None):
        return inputs

    def step(self, inputs, states):
        # Split the hidden state into blocks (each U, V, W are shared across blocks).
        state = tf.split(states[0], self._num_blocks, axis=1)
        print('state after split', state)

        next_states = []
        for j, state_j in enumerate(state):  # Hidden State (j)
            key_j = tf.expand_dims(self._keys[j], axis=0)
            gate_j = self.get_gate(state_j, key_j, inputs)
            candidate_j = self.get_candidate(state_j, key_j, inputs, self.U, self.V, self.W, self.U_bias)

            # Equation 4: h_j <- h_j + g_j * h_j^~
            # Perform an update of the hidden state (memory).
            state_j_next = state_j + tf.expand_dims(gate_j, -1) * candidate_j

            # Equation 5: h_j <- h_j / \norm{h_j}
            # Forget previous memories by normalization.
            state_j_next_norm = tf.norm(
                tensor=state_j_next,
                ord='euclidean',
                axis=-1,
                keep_dims=True)
            state_j_next_norm = tf.where(
                tf.greater(state_j_next_norm, 0.0),
                state_j_next_norm,
                tf.ones_like(state_j_next_norm))
            state_j_next = state_j_next / state_j_next_norm
            next_states.append(state_j_next)
        state_next = tf.concat(next_states, axis=1)
        return state_next, [state_next]

    def get_constants(self, inputs, training=None):
        return []

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = self.zero_state(self.initial_batch_size)
        return initial_state


class RENMask(Layer):
    '''Apply mask to data'''
    # Initialise the parameters
    def __init__(self, embedding_size, vocab_size, sentence_len, **kwargs):
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.sentence_len = sentence_len
        self.supports_masking = True
        self.initializer = tf.random_normal_initializer(stddev=0.1)

        super(RENMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialise trainable weights
        self.Mask = self.add_weight((self.sentence_len, self.embedding_size), initializer=tf.constant_initializer(1.0), name='Mask',
                                    trainable=True)

        self.supports_masking = True
        super(RENMask, self).build(input_shape)

    def call(self, x):
        embeddings = tf.multiply(x, self.Mask)
        embeddings = tf.reduce_sum(embeddings, axis=[2])

        return embeddings


class RENEmbed(Layer):
    '''Layer for word embeddings'''
    # Initialise the parameters
    def __init__(self, embedding_size, vocab_size, sentence_len, **kwargs):
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.sentence_len = sentence_len
        self.supports_masking = True
        self.initializer = tf.random_normal_initializer(stddev=0.1)

        super(RENEmbed, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialise trainable weights
        self.E = self.add_weight((self.vocab_size, self.embedding_size), initializer=self.initializer, name='E', trainable=True)

        zero_mask = tf.constant(
            value=[0 if i == 0 else 1 for i in range(self.vocab_size)],
            shape=[self.vocab_size, 1],
            dtype=tf.float32)

        self.E = self.E * zero_mask

        self.supports_masking = True
        super(RENEmbed, self).build(input_shape)

    def call(self, x):
        embeddings = tf.nn.embedding_lookup(self.E, x)
        return embeddings


class RENPred(Layer):
    # Initialise the parameters
    def __init__(self, axis=-1, **kwargs):
        self._axis = axis
        super(RENPred, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RENPred, self).build(input_shape)

    def call(self, x):
        print('x', x, type(x))
        preds = K.argmax(x, axis=self._axis)
        return preds