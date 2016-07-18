from keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint
)
from keras.datasets import cifar100
from keras.layers import (
    Activation,
    Input,
    Dense,
    Flatten
)
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam#, Nadam
from keras.utils.visualize_util import plot
from keras.utils import np_utils

from fractalnet import fractal_net

NB_CLASSES = 100
NB_EPOCHS = 400
LEARN_START = 0.02
BATCH_SIZE = 100
MOMENTUM = 0.9

(X_train, y_train), (X_test, y_test) = cifar100.load_data()

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print X_train.shape

# Drop by 10 when we halve the number of remaining epochs (200, 300, 350, 375)
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
    dropout = [0., 0.1, 0.2, 0.3, 0.4]
    conv = [(64, 3, 3), (128, 3, 3), (256, 3, 3), (512, 3, 3), (512, 2, 2)]
    input= Input(shape=(3, 32, 32))
    output = fractal_net(
        c=3, b=5, conv=conv,
        drop_path=0.15, dropout=dropout)(input)
    output = Flatten()(output)
    output = Dense(NB_CLASSES)(output)
    output = Activation('softmax')(output)
    model = Model(input=input, output=output)
    optimizer = SGD(lr=LEARN_START, momentum=MOMENTUM)
    #optimizer = RMSprop(lr=LEARN_START)
    #optimizer = Adam()
    #optimizer = Nadam()
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
        #callbacks=[snapshot]
    )

net = build_network()
train_network(net)
