import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def simpleGAN_build_generator(input_shape, img_len):
    layer_input = tf.keras.Input(input_shape)
    layers = tf.keras.layers.Flatten() (layer_input)
    layers = tf.keras.layers.Dense(256) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dense(512) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dense(1024) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dense(img_len, activation='tanh') (layers)

    model = tf.keras.models.Model(layer_input, layers)
    return model


def simpleGAN_build_discriminator(img_len):
    layer_input = tf.keras.Input(img_len)
    layers = tf.keras.layers.Flatten() (layer_input)
    layers = tf.keras.layers.Dense(1024) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dropout(rate=0.5) (layers)
    layers = tf.keras.layers.Dense(512) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dense(256) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dense(1, activation='sigmoid') (layers)
    #layers = tf.keras.layers.Reshape((img_len/2, img_len/2)) (layers)

    model = tf.keras.models.Model(layer_input, layers)
    return model
