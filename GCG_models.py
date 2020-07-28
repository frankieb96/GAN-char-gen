import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def simpleGAN_build_generator(latent_dimension=100, name='SimpleGAN_generator', img_side=28):
    """
    # TODO write pydocs

    :param latent_dimension:
    :return:
    """
    layer_input = tf.keras.Input(latent_dimension)

    layers = tf.keras.layers.Dense(256)(layer_input)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = tf.keras.layers.Dense(1024)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = tf.keras.layers.Dense(img_side ** 2, activation='tanh')(layers)
    layers = tf.keras.layers.Reshape((img_side, img_side))(layers)

    model = tf.keras.models.Model(layer_input, layers, name=name)
    return model


def simpleGAN_build_discriminator(img_shape=(28, 28), name='SimpleGAN_discriminator'):
    """
    # TODO write pydocs

    :param img_shape:
    :return:
    """
    layer_input = tf.keras.Input(img_shape)

    layers = tf.keras.layers.Flatten()(layer_input)
    layers = tf.keras.layers.Dense(1024)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dropout(rate=0.5)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dropout(rate=0.5)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = tf.keras.layers.Dense(256)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dropout(rate=0.5)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = tf.keras.layers.Dense(1, activation='sigmoid')(layers)

    model = tf.keras.models.Model(layer_input, layers, name=name)
    return model


def DCGAN_build_generator(latent_dimension=100, name='DCGAN_generator'):
    """
    # TODO write pydocs

    :param latent_dimension:
    :param name:
    :param img_side:
    :return:
    """
    layer_input = tf.keras.Input(latent_dimension)

    layers = tf.keras.layers.Dense(7 * 7 * 128)(layer_input)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Reshape([7, 7, 128])(layers)
    layers = tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', activation='tanh')(layers)

    model = tf.keras.models.Model(layer_input, layers, name=name)
    return model


def DCGAN_build_discriminator(img_shape=(28, 28, 1), name='DCGAN_discriminator'):
    """
    # TODO write pydocs

    :param img_shape:
    :return:
    """
    layer_input = tf.keras.Input(img_shape)
    layers = tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same')(layer_input)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dropout(rate=0.4)(layers)
    layers = tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dropout(rate=0.4)(layers)
    layers = tf.keras.layers.Flatten()(layers)
    layers = tf.keras.layers.Dense(1, activation='sigmoid')(layers)
    model = tf.keras.models.Model(layer_input, layers, name=name)
    return model


def AAE_build_encoder(img_shape=(28, 28), latent_dim=100, name='AAE_encoder'):
    input_layer = tf.keras.Input(img_shape)

    layers = tf.keras.layers.Flatten()(input_layer)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(2 * latent_dim)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)

    model = tf.keras.Model(input_layer, layers, name=name)
    return model


def AAE_build_decoder(latent_dim=100, img_side=28, name='AAE_decoder'):
    input_layer = tf.keras.Input(latent_dim)

    layers = tf.keras.layers.Flatten()(input_layer)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(img_side ** 2)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Reshape((img_side, img_side))(layers)
    layers = tf.keras.layers.Activation('tanh')(layers)

    model = tf.keras.Model(input_layer, layers, name=name)
    return model


def AAE_build_discriminator(latent_dim=100, name='AAE_discriminator'):
    input_layer = tf.keras.Input(latent_dim)

    layers = tf.keras.layers.Flatten()(input_layer)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(256)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(1)(layers)
    layers = tf.keras.layers.Activation('sigmoid')(layers)

    model = tf.keras.Model(input_layer, layers, name=name)
    return model
