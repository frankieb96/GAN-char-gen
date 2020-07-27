import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def simpleGAN_build_generator(input_shape, img_len):
    layer_input = tf.keras.Input(input_shape)
    layers = tf.keras.layers.Flatten() (layer_input)
    layers = tf.keras.layers.Dense(256, input_dim=input_shape) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dense(512) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dense(1024) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dense(img_len, activation='tanh') (layers)

    model = tf.keras.models.Model(layer_input, layers)
    return model


def simpleGAN_build_discriminator(input_shape, img_shape):
    layer_input = tf.keras.Input(input_shape)
    layers = tf.keras.layers.Flatten() (layer_input)
    layers = tf.keras.layers.Dense(1024, input_dim=input_shape) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dropout(rate=0.5) (layers)
    layers = tf.keras.layers.Dense(512) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dense(256) (layers) (layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2) (layers)
    layers = tf.keras.layers.Dense(1, activation='sigmoid') (layers)

    model = tf.keras.models.Model(layer_input, layers)
    return model
