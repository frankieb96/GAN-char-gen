import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def simpleGAN_build_generator(latent_dimension, name='SimpleGAN_generator', img_side=28):
    """
    # TODO write pydocs

    :param latent_dimension:
    :return:
    """
    layer_input = tf.keras.Input(latent_dimension)

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
    layers = tf.keras.layers.Dense(img_side**2, activation='tanh') (layers)
    layers = tf.keras.layers.Reshape((img_side, img_side)) (layers)

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
