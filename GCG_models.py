import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def simpleGAN_build_generator(input_shape, name='SimpleGAN_generator'):
    """
    # TODO write pydocs

    :param input_shape:
    :return:
    """
    img_len = input_shape[0]*input_shape[1]
    layer_input = tf.keras.Input(input_shape)

    layers = tf.keras.layers.Flatten() (layer_input)
    layers = tf.keras.layers.Dense(256) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.BatchNormalization() (layers)
    layers = tf.keras.layers.Dense(512) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.BatchNormalization() (layers)
    layers = tf.keras.layers.Dense(1024) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.BatchNormalization() (layers)
    layers = tf.keras.layers.Dense(img_len, activation='tanh') (layers)
    layers = tf.keras.layers.Reshape((28, 28)) (layers)

    model = tf.keras.models.Model(layer_input, layers, name=name)
    return model


def simpleGAN_build_discriminator(img_shape, name='SimpleGAN_discriminator'):
    """
    # TODO write pydocs

    :param img_shape:
    :return:
    """
    layer_input = tf.keras.Input(img_shape)

    layers = tf.keras.layers.Flatten() (layer_input)
    layers = tf.keras.layers.Dense(1024) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dropout(rate=0.5) (layers)
    tf.keras.layers.BatchNormalization() (layers)
    layers = tf.keras.layers.Dense(512) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dropout(rate=0.5) (layers)
    tf.keras.layers.BatchNormalization() (layers)
    layers = tf.keras.layers.Dense(256) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dropout(rate=0.5) (layers)
    tf.keras.layers.BatchNormalization() (layers)
    layers = tf.keras.layers.Dense(1, activation='sigmoid') (layers)

    model = tf.keras.models.Model(layer_input, layers, name=name)
    return model
