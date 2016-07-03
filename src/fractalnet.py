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
def one_one_in_array(count, seed=None):
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
    def __init__(self, drop_p, is_global, global_path, **kwargs):
        #print "init"
        self.p = 1. - drop_p
        self.is_global = is_global
        self.global_path = global_path
        self.uses_learning_phase = True
        super(JoinLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #print("build")
        pass

    def random_arr(self, count, p):
        return K.random_binomial((count,), p=p)

    def arr_with_one(self, count):
        return one_one_in_array(count=count)

    def _gen_local_drops(self, count, p):
        arr = self.random_arr(count, p)
        drops = K.switch(
            K.any(arr),
            arr,
            self.arr_with_one(count)
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
        ave = K.variable(0.)
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

    def ave(self, inputs):
        ave = inputs[0]
        for input in inputs[1:]:
            ave += input
        ave /= len(inputs)
        return ave

    def call(self, inputs, mask=None):
        print("call")
        print(K.shape(inputs[0]))
        output = K.in_train_phase(self._drop_path(inputs), self.ave(inputs))
        return output

    def get_output_shape_for(self, input_shape):
        print("get_output_shape_for", input_shape)
        return input_shape[0]

class JoinLayerGen:
    '''JoinLayerGen will initialize seeds for both the global dropout
    switch and for the global droppout path.

    These seeds will be used to create the random tensors that the
    children layers will use to know if they should use global
    droppout and which path to take in case it's choosen.
    '''
    def __init__(self, width, global_p=0.5):
        self.global_p = global_p
        self.width = width
        self.switch_seed = np.random.randint(1, 10e6)
        self.path_seed = np.random.randint(1, 10e6)
        self.path_array = self._build_global_path_arr()
        self.is_global = self._build_global_switch()

    def _build_global_path_arr(self):
        # The path the block will take when using global droppath
        return one_one_in_array(seed=self.path_seed, count=self.width)

    def _build_global_switch(self):
        # A random tensor that will signal if the batch should
        # use global or local droppath
        return K.equal(K.random_binomial((), p=self.global_p, seed=self.switch_seed), 1.)

    def get_join_layer(self, drop_p):
        global_switch = self.is_global
        global_path = self.path_array
        return JoinLayer(drop_p=drop_p, is_global=global_switch, global_path=global_path)

def fractal_conv(prev, filter, nb_row, nb_col, dropout=None):
    print(prev)
    conv = prev
    conv = Convolution2D(filter, nb_row=nb_col, nb_col=nb_col, init='glorot_normal', border_mode='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    if dropout:
        conv = Dropout(dropout)(conv)
    return conv

def fractal_block_iter(join_gen, z, c, filter, nb_col, nb_row, drop_p, dropout=None):
    columns = [[z] for _ in range(c)]
    for row in range(2**(c-1)):
        t_row = []
        for col in range(c):
            prop = 2**(col)
            # Add blocks
            if (row+1) % prop == 0:
                t_col = columns[col]
                t_col.append(fractal_conv(t_col[-1], filter=filter, nb_col=nb_col, nb_row=nb_row, dropout=dropout))
                t_row.append(col)
        # Merge (if needed)
        if len(t_row) > 1:
            merging = [columns[x][-1] for x in t_row]
            merged  = join_gen.get_join_layer(drop_p=drop_p)(merging)
            for i in t_row:
                columns[i].append(merged)
    return columns[0][-1]

def fractal_iter(z, b, c, conv, drop_path, global_p=0.5, dropout=None):
    input = z
    join_gen = JoinLayerGen(width=c, global_p=global_p)
    for i in range(b):
        (filter, nb_col, nb_row) = conv[i]
        dropout_i = dropout[i] if dropout else None
        input = fractal_block_iter(join_gen=join_gen,
                                   z=input, c=c,
                                   filter=filter,
                                   nb_col=nb_col,
                                   nb_row=nb_row,
                                   drop_p=drop_path,
                                   dropout=dropout_i)
        input = MaxPooling2D(pool_size=(2,2))(input)
    return input

def fractal_iter_test(c, b):
    input = Input(shape=(255, 32, 32))
    net = fractal_iter(input, b=1, c=4, conv=[32], drop_path=0.1)
    model = Model(input=input, output=net)
    plot(model, to_file='model.png')

#fractal_rec(4)
#net = fractal_iter_test(4, 1)
