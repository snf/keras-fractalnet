import random, string
import numpy as np
from keras.layers import (
    Input,
    BatchNormalization,
    Activation, Dense, Merge,
    Convolution2D, MaxPooling2D, ZeroPadding2D
)
from keras.models import Model
from keras.engine import Layer
from keras.utils.visualize_util import plot
from keras import backend as K
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.tensor.shared_randomstreams import RandomStreams

class JoinLayer(Layer):
    def __init__(self, drop_p, **kwargs):
        #print "init"
        self.p = 1. - drop_p
        super(JoinLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #print("build")
        pass

    def random_arr(self, count, p):
        return K.random_binomial((count,), p=p)

    def arr_with_one(self, count):
        pvals = [[1.0/count for _ in range(count)]]
        rng = RandomStreams()
        arr = rng.multinomial(n=1, pvals=pvals, dtype='float32')[0]
        return arr

    def _gen_drops(self, count, p):
        arr = self.random_arr(count, p)
        drops = K.switch(
            K.not_equal(arr.max(), 0),
            arr,
            self.arr_with_one(count)
        )
        return drops

    def _drop_path(self, inputs):
        count = len(inputs)
        drops = self._gen_drops(count, self.p)

        ave = K.variable(0)
        for i in range(0, count):
            ave += inputs[i] * drops[i]
        ave /= drops.sum()
        return ave

    def drop_path(self, inputs):
        drops = []
        # Drop
        for i in range(len(inputs)):
            if np.random.rand() < self.p:
                drops.append(i)
        # Be sure there is at least one alive
        if len(drops) == 0:
            alive = np.random.randint(0, len(inputs))
            drops.append(alive)
        # Now average
        ave = inputs[drops[0]]
        for i in drops[1:]:
            ave += inputs[i]
        ave /= len(drops)
        return ave

    def ave(self, inputs):
        ave = inputs[0]
        for input in inputs[1:]:
            ave += input
        ave /= len(inputs)
        return ave

    def call(self, inputs, mask=None):
        print("call")
        print inputs[0].shape
        output = K.in_train_phase(self._drop_path(inputs), self.ave(inputs))
        return output

    def get_output_shape_for(self, input_shape):
        print("get_output_shape_for", input_shape)
        return input_shape[0]

def fractal_conv(prev, filter, dropout=None):
    print prev
    conv = prev
    conv = ZeroPadding2D((1,1))(conv)
    conv = Convolution2D(filter, 3, 3, init='glorot_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv

def fractal_merge(prev, drop_p):
    merge = JoinLayer(drop_p=drop_p)(prev)
    return merge

def fractal_block_iter(z, c, filter, drop_p, dropout=None):
    columns = [[z] for _ in range(c)]
    p_columns = []
    for row in range(2**(c-1)):
        p_t_row = []
        t_row = []
        for col in range(c):
            # print('row:', row, ', col:', col)
            prop = 2**(col)
            # Add blocks
            if (row+1) % prop == 0:
                t_col = columns[col]
                t_col.append(fractal_conv(t_col[-1], filter, dropout=dropout))
                t_row.append(col)
                p_t_row.append('x')
            else:
                p_t_row.append('o')
        p_columns.append(p_t_row)
        # Merge (if needed)
        if len(t_row) > 1:
            merging = [columns[x][-1] for x in t_row]
            merged  = fractal_merge(merging, drop_p=drop_p)
            for i in t_row:
                columns[i].append(merged)
        p_convs = [e for e in p_t_row if e == 'x']
        if len(p_convs) > 1:
            p_merge_row = ['=' for i in range(len(p_convs))]
            p_merge_row+= [' ' for i in range(len(p_t_row) - len(p_convs))]
            p_columns.append(p_merge_row)

    # for i in p_columns:
    #     print(i)
    return columns[0][-1]

def fc(z, c):
    conv_layer = fractal_conv(z, 64)
    if c == 1:
        return conv_layer
    else:
        this_conv = fc(fc(z, c-1), c-1)
        return Merge(mode='ave')([this_conv, conv_layer])

def fractal_rec(c):
    input = Input(shape=(3,64,64))
    output= fc(input, c)
    model = Model(input=input, output=output)
    plot(model, to_file='model.png')
    return model

def fractal_iter(z, b, c, conv, drop_path, dropout=None):
    input = z
    for i in range(b):
        filter = conv[i]
        input = fractal_block_iter(z=input, c=c,
                                   filter=filter,
                                   drop_p=drop_path,
                                   dropout=dropout)
        input = MaxPooling2D()(input)
    return input

def fractal_iter_test(c, b):
    input = Input(shape=(255, 32, 32))
    net = fractal_iter(input, b=1, c=4, conv=[32], drop_path=0.1)
    model = Model(input=input, output=net)
    plot(model, to_file='model.png')

#fractal_rec(4)
#net = fractal_iter_test(4, 1)
