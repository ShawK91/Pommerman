import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

os.environ["CUDA_VISIBLE_DEVICES"]='3'

#KERAS
def load_keras_model(model_file):
    return load_model(model_file)

def init_model():
    c = x = Input(shape=(11, 11, 18))
    c = Conv2D(filters=32, kernel_size=3, strides=1, activation="elu", padding="same")(c)
    c = Conv2D(filters=32, kernel_size=3, strides=1, activation="elu", padding="same")(c)
    #c = Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same")(c)
    h = Flatten()(c)
    h = Dense(128, activation='elu')(h)
    p = Dense(6, activation="softmax", name='p')(h)
    v = Dense(1, activation="tanh", name='v')(h)
    model = Model(x, [p, v])
    model.summary()
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mse'], loss_weights=[1, 0.1],
                  metrics={'p': 'accuracy'})
    return model