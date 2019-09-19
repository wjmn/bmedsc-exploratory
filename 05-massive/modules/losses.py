
import tensorflow as tf
import numpy as np
from functools import partial

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
categorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy()


def encoder_loss_basic(real_labels: tf.Tensor, decoded_one_hot: tf.Tensor, *args, **kwargs) -> tf.Tensor:
    real_one_hot = tf.one_hot(real_labels, depth=11)
    label_loss = categorical_cross_entropy(real_one_hot, decoded_one_hot)
    return label_loss


def encoder_loss_mean_all_cosine(real_labels: tf.Tensor, decoded_one_hot: tf.Tensor, renders, *args, **kwargs) -> tf.Tensor:
    """ Mean of the cosine (pi/2-scaled) values of each render i.e. pixel-wise.
    """
    real_one_hot = tf.one_hot(real_labels, depth=11)
    label_loss = categorical_cross_entropy(real_one_hot, decoded_one_hot)
    brightness_loss = tf.reduce_mean(tf.math.cos(tf.constant(np.pi / 2) * renders))
    return label_loss + brightness_loss


def encoder_loss_mean_sum_cosine(real_labels: tf.Tensor, decoded_one_hot: tf.Tensor, renders, *args, **kwargs) -> tf.Tensor:
    """ Mean of the summed cosine (pi/2-scaled) values of each render.
    """
    real_one_hot = tf.one_hot(real_labels, depth=11)
    label_loss = categorical_cross_entropy(real_one_hot, decoded_one_hot)
    brightness_loss = tf.reduce_mean(tf.reduce_sum(tf.math.cos(tf.constant(np.pi / 2) * renders), axis=0))
    return label_loss + brightness_loss


def encoder_loss_sum_all_cosine(real_labels: tf.Tensor, decoded_one_hot: tf.Tensor, renders, *args, **kwargs) -> tf.Tensor:
    """ Sum of the cosine (pi/2-scaled) values of each render (i.e. pixel-wise).
    """
    real_one_hot = tf.one_hot(real_labels, depth=11)
    label_loss = categorical_cross_entropy(real_one_hot, decoded_one_hot)
    brightness_loss = tf.reduce_sum(tf.math.cos(tf.constant(np.pi / 2) * renders))
    return label_loss + brightness_loss


DICT_LOSS_ENCODER = {
    'basic': encoder_loss_basic,
    'mean-all-cosine': encoder_loss_mean_all_cosine,
    'mean-sum-cosine': encoder_loss_mean_sum_cosine,
    'sum-all-cosine': encoder_loss_sum_all_cosine,
}


def decoder_loss_base(real_labels, decoded_real_one_hot, decoded_fake_one_hot, real_weight, fake_weight, *args, **kwargs) -> tf.Tensor:
    shape = real_labels.shape

    real_one_hot = tf.one_hot(real_labels, depth=11)
    real_loss = categorical_cross_entropy(real_one_hot, decoded_real_one_hot)

    all_fake_one_hot = tf.one_hot(tf.fill(shape, 10), depth=11)
    fake_loss = categorical_cross_entropy(all_fake_one_hot, decoded_fake_one_hot)

    loss = (real_loss * real_weight) + (fake_loss * fake_weight)

    return loss


DICT_LOSS_DECODER = {
    'basic': partial(decoder_loss_base, real_weight=1, fake_weight=1),
    'real-weighted': partial(decoder_loss_base, real_weight=10, fake_weight=1),
    'fake-weighted': partial(decoder_loss_base, real_weight=1, fake_weight=10)
}
