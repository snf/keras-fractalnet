import numpy as np
from keras.layers import (
    Input,
    BatchNormalization,
    Activation, Dense, Dropout, Merge,
    Convolution2D, MaxPooling2D, ZeroPadding2D
)
from keras.models import Model
from keras.engine import Layer
from keras.utils.visualize_util import plot
from keras import backend as K

if K._BACKEND == 'theano':
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
if K._BACKEND == 'tensorflow':
    import tensorflow as tf

def theano_multinomial(n, pvals, seed):
    rng = RandomStreams(seed)
    return rng.multinomial(n=n, pvals=pvals, dtype='float32')

def tensorflow_categorical(count, seed):
    assert count > 0
    arr = [1.] + [.0 for _ in range(count-1)]
    return tf.random_shuffle(arr, seed)

# Returns a random array [x0, x1, ...xn] where one is 1 and the others
# are 0. Ex: [0, 0, 1, 0].
def rand_one_in_array(count, seed=None):
    if seed is None:
        seed = np.random.randint(1, 10e6)
    if K._BACKEND == 'theano':
        pvals = np.array([[1. / count for _ in range(count)]], dtype='float32')
        return theano_multinomial(n=1, pvals=pvals, seed=seed)[0]
    elif K._BACKEND == 'tensorflow':
        return tensorflow_categorical(count=count, seed=seed)
    else:
        raise Exception('Backend: {} not implemented'.format(K._BACKEND))

class JoinLayer(Layer):
    '''
    This layer will behave as Merge(mode='ave') during testing but
    during training it will randomly select between using local or
    global droppath and apply the average of the paths alive after
    aplying the drops.

    - Global: use the random shared tensor to select the paths.
    - Local: sample a random tensor to select the paths.
    '''

    def __init__(self, drop_p, is_global, global_path, force_path, **kwargs):
        #print "init"
        self.p = 1. - drop_p
        self.is_global = is_global
        self.global_path = global_path
        self.uses_learning_phase = True
        self.force_path = force_path
        super(JoinLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #print("build")
        self.average_shape = list(input_shape[0])[1:]

    def _random_arr(self, count, p):
        return K.random_binomial((count,), p=p)

    def _arr_with_one(self, count):
        return rand_one_in_array(count=count)

    def _gen_local_drops(self, count, p):
        # Create a local droppath with at least one path
        arr = self._random_arr(count, p)
        drops = K.switch(
            K.any(arr),
            arr,
            self._arr_with_one(count)
        )
        return drops

    def _gen_global_path(self, count):
        return self.global_path[:count]

    def _drop_path(self, inputs):
        count = len(inputs)
        drops = K.switch(
            self.is_global,
            self._gen_global_path(count),
            self._gen_local_drops(count, self.p)
        )
        ave = K.zeros(shape=self.average_shape)
        for i in range(0, count):
            ave += inputs[i] * drops[i]
        sum = K.sum(drops)
        # Check that the sum is not 0 (global droppath can make it
        # 0) to avoid divByZero
        ave = K.switch(
            K.not_equal(sum, 0.),
            ave/sum,
            ave)
        return ave

    def _ave(self, inputs):
        ave = inputs[0]
        for input in inputs[1:]:
            ave += input
        ave /= len(inputs)
        return ave

    def call(self, inputs, mask=None):
        #print("call")
        if self.force_path:
            output = self._drop_path(inputs)
        else:
            output = K.in_train_phase(self._drop_path(inputs), self._ave(inputs))
        return output

    def get_output_shape_for(self, input_shape):
        #print("get_output_shape_for", input_shape)
        return input_shape[0]

class JoinLayerGen:
    '''
    JoinLayerGen will initialize seeds for both global droppath
    switch and global droppout path.

    These seeds will be used to create the random tensors that the
    children layers will use to know if they must use global droppout
    and which path to take in case it is.
    '''

    def __init__(self, width, global_p=0.5, deepest=False):
        self.global_p = global_p
        self.width = width
        self.switch_seed = np.random.randint(1, 10e6)
        self.path_seed = np.random.randint(1, 10e6)
        self.deepest = deepest
        if deepest:
            self.is_global = K.variable(1.)
            self.path_array = K.variable([1.] + [.0 for _ in range(width-1)])
        else:
            self.is_global = self._build_global_switch()
            self.path_array = self._build_global_path_arr()

    def _build_global_path_arr(self):
        # The path the block will take when using global droppath
        return rand_one_in_array(seed=self.path_seed, count=self.width)

    def _build_global_switch(self):
        # A randomly sampled tensor that will signal if the batch
        # should use global or local droppath
        return K.equal(K.random_binomial((), p=self.global_p, seed=self.switch_seed), 1.)

    def get_join_layer(self, drop_p):
        global_switch = self.is_global
        global_path = self.path_array
        return JoinLayer(drop_p=drop_p, is_global=global_switch, global_path=global_path, force_path=self.deepest)

def fractal_conv(filter, nb_row, nb_col, dropout=None):
    def f(prev):
        conv = prev
        conv = Convolution2D(filter, nb_row=nb_col, nb_col=nb_col, init='he_normal', border_mode='same')(conv)
        if dropout:
            conv = Dropout(dropout)(conv)
        conv = BatchNormalization(mode=0, axis=1)(conv)
        conv = Activation('relu')(conv)
        return conv
    return f

# XXX_ It's not clear when to apply Dropout, the paper cited
# (arXiv:1511.07289) uses it in the last layer of each stack but in
# the code gustav published it is in each convolution block so I'm
# copying it.
def fractal_block(join_gen, c, filter, nb_col, nb_row, drop_p, dropout=None):
    def f(z):
        columns = [[z] for _ in range(c)]
        last_row = 2**(c-1) - 1
        for row in range(2**(c-1)):
            t_row = []
            for col in range(c):
                prop = 2**(col)
                # Add blocks
                if (row+1) % prop == 0:
                    t_col = columns[col]
                    t_col.append(fractal_conv(filter=filter,
                                              nb_col=nb_col,
                                              nb_row=nb_row,
                                              dropout=dropout)(t_col[-1]))
                    t_row.append(col)
            # Merge (if needed)
            if len(t_row) > 1:
                merging = [columns[x][-1] for x in t_row]
                merged  = join_gen.get_join_layer(drop_p=drop_p)(merging)
                for i in t_row:
                    columns[i].append(merged)
        return columns[0][-1]
    return f

def fractal_net(b, c, conv, drop_path, global_p=0.5, dropout=None, deepest=False):
    '''
    Return a function that builds the Fractal part of the network
    respecting keras functional model.
    When deepest is set, we build the entire network but set droppath
    to global and the Join masks to [1., 0... 0.] so only the deepest
    column is always taken.
    We don't add the softmax layer here nor build the model.
    '''
    def f(z):
        output = z
        # Initialize a JoinLayerGen that will be used to derive the
        # JoinLayers that share the same global droppath
        join_gen = JoinLayerGen(width=c, global_p=global_p, deepest=deepest)
        for i in range(b):
            (filter, nb_col, nb_row) = conv[i]
            dropout_i = dropout[i] if dropout else None
            output = fractal_block(join_gen=join_gen,
                                   c=c, filter=filter,
                                   nb_col=nb_col,
                                   nb_row=nb_row,
                                   drop_p=drop_path,
                                   dropout=dropout_i)(output)
            output = MaxPooling2D(pool_size=(2,2), strides=(2,2))(output)
        return output
    return f
