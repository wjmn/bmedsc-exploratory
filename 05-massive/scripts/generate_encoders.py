import tensorflow as tf
import json
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import *
from typing import List
from datetime import datetime

# Setting this for the test run. Probably should put an assertion in the train.py
# to check whether the encoder/decoder and grid sizes match, but will leave it like
# this for now.
NUM_PHOSPHENES = 8 * 8


def encoder_basic():
    input_labels = Input(shape=(1,))
    layer_labels = Embedding(10, 10)(input_labels)
    layer_labels = Flatten()(layer_labels)
    layer_labels = BatchNormalization()(layer_labels)
    layer_labels = LeakyReLU()(layer_labels)
    layer_labels = Dense(NUM_PHOSPHENES, activation=tf.nn.sigmoid)(layer_labels)
    return Model(input_labels, layer_labels)


def encoder_double():
    input_labels = Input(shape=(1,))
    layer_labels = Embedding(10, 10)(input_labels)
    layer_labels = Flatten()(layer_labels)
    layer_labels = BatchNormalization()(layer_labels)
    layer_labels = LeakyReLU()(layer_labels)
    layer_labels = Dense(NUM_PHOSPHENES)(layer_labels)
    layer_labels = LeakyReLU()(layer_labels)
    layer_labels = Dense(NUM_PHOSPHENES, activation=tf.nn.sigmoid)(layer_labels)
    return Model(input_labels, layer_labels)


def encoder_direct():
    input_labels = Input(shape=(1,))
    layer_labels = Dense(NUM_PHOSPHENES)(input_labels)
    layer_labels = LeakyReLU()(layer_labels)
    layer_labels = Dense(NUM_PHOSPHENES, activation=tf.nn.sigmoid)(layer_labels)
    return Model(input_labels, layer_labels)


ENCODERS = [
    encoder_basic,
    encoder_double,
    encoder_direct,
]


for encoder in ENCODERS:
    encoder_id = 'E-' + datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S-%f')
    print(encoder_id)
    encoder_json = encoder().to_json()
    with open('../models/json/encoders/' + encoder_id + '.json', 'w') as outfile:
        outfile.write(encoder_json)
