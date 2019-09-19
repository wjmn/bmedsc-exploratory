import tensorflow as tf
import json
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import *
from typing import List
from datetime import datetime

# Setting this for the test run. Probably should put an assertion in the train.py
# to check whether the encoder/decoder and grid sizes match, but will leave it like
# this for now.
X_RENDER = 48
Y_RENDER = 48


def decoder_basic():
    input = Input(shape=(Y_RENDER, X_RENDER))
    layer = Reshape((Y_RENDER, X_RENDER, 1))(input)
    layer = Conv2D(64, (8,8), padding='same', strides=(2,2))(layer)
    layer = LeakyReLU()(layer)
    layer = Dropout(0.25)(layer)
    layer = Conv2D(128, (4,4), padding='same', strides=(2,2))(layer)
    layer = LeakyReLU()(layer)
    layer = Dropout(0.25)(layer)
    layer = Flatten()(layer)
    layer = Dense(11, activation=tf.nn.softmax)(layer)
    return Model(inputs=input, outputs=layer)


def decoder_deeper():
    input = Input(shape=(Y_RENDER, X_RENDER))
    layer = Reshape((Y_RENDER, X_RENDER, 1))(input)
    layer = Conv2D(32, (12,12), padding='same', strides=(2,2))(layer)
    layer = LeakyReLU()(layer)
    layer = Dropout(0.25)(layer)
    layer = Conv2D(64, (8,8), padding='same', strides=(2,2))(layer)
    layer = LeakyReLU()(layer)
    layer = Dropout(0.25)(layer)
    layer = Conv2D(128, (4,4), padding='same', strides=(2,2))(layer)
    layer = LeakyReLU()(layer)
    layer = Dropout(0.25)(layer)
    layer = Flatten()(layer)
    layer = Dense(11, activation=tf.nn.softmax)(layer)
    return Model(inputs=input, outputs=layer)


DECODERS = [
    decoder_basic,
    decoder_deeper,
]


for decoder in DECODERS:
    decoder_id = 'D-' + datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S-%f')
    print(decoder_id)
    decoder_json = decoder().to_json()
    with open('../models/json/decoders/' + decoder_id + '.json', 'w') as outfile:
        outfile.write(decoder_json)
