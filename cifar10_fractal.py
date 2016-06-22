from keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint
)
from keras.datasets import cifar10
from keras.layers import (
    Activation,
    Input,
    Dense,
    Flatten
)
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from keras.utils import np_utils

from fractalnet import fractal_iter

NB_CLASSES = 10
NB_EPOCHS = 400
# Drop by 10 when we halve the number of remaining epochs (200, 300, 350, 375)
LEARN_START = 0.02
BATCH_SIZE = 100
MOMENTUM = 0.9
INITIALIZE = 'xavier'

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
print X_train.shape

def learning_rate(epoch):
    if epoch < 200:
        return 0.02
    if epoch < 300:
        return 0.002
    if epoch < 350:
        return 0.0002
    if epoch < 375:
        return 0.00002
    return 0.000002

def build_network():
    #dropout = [0, 0.1, 0.2, 0.3, 0.4]
    dropout = None
    conv = [64, 128, 256, 512, 512]
    input= Input(shape=(3, 32, 32))
    net = fractal_iter(
        z=input,
        c=3, b=5, conv=conv,
        drop_path=0.15, dropout=dropout)
    output = Flatten()(net)
    output = Dense(NB_CLASSES)(output)
    output = Activation('softmax')(output)
    model = Model(input=input, output=output)
    optimizer = SGD(lr=LEARN_START, momentum=MOMENTUM)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    plot(model, to_file='model.png')
    return model

def train_network(net):
    snapshot = ModelCheckpoint(
        filepath="snapshots/weights.{epoch:04d}-{val_loss:.4f}.h5",
        monitor="val_loss",
        save_best_only=False)
    learn = LearningRateScheduler(learning_rate)
    net.fit(
        x=X_train, y=Y_train, batch_size=BATCH_SIZE,
        nb_epoch=NB_EPOCHS, validation_data=(X_test, Y_test),
        callbacks=[learn, snapshot]
    )

net = build_network()
train_network(net)
