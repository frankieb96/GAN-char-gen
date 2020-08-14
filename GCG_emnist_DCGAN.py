"""
Author: Francesco Bianco
Student number: 1234358
Email: francesco.bianco.5@studenti.unipd.it

Program file GCG_emnist_DCGAN.py
"""

import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import GCG_utils

""" GENERATOR/DISCRIMINATOR MODEL CREATOR FUNCTIONS """


def DCGAN_build_generator(latent_dimension=100, name='EMNIST_DCGAN_generator'):
    layer_input = tf.keras.Input(latent_dimension)

    layers = tf.keras.layers.Dense(7 * 7 * 128)(layer_input)
    layers = tf.keras.layers.Reshape([7, 7, 128])(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', activation='selu',
                                             kernel_initializer='lecun_normal')(layers)
    layers = tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', activation='tanh')(layers)

    # the output has shape (28, 28, 1)
    model = tf.keras.models.Model(layer_input, layers, name=name)
    return model


def DCGAN_build_discriminator(img_shape=(28, 28, 1), name='EMNIST_DCGAN_discriminator'):
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


""" ========================================================================================================== """

tf.random.set_seed(1)
latent_dimension = 100
img_shape = (28, 28, 1)
batch_size = 32
n_epochs = 30
if len(sys.argv) < 3:
    print("WARNING: not enough input params. Resorting to default params. Usage is 'net-name' 'dataset'",
          file=sys.stderr)
    NAME = "E_DCGAN"
    DATASET = "EMNIST"
else:
    NAME = sys.argv[1]
    DATASET = sys.argv[2]
PATH = "temp_project/{}/".format(NAME)
print("PARAMS are: ", [NAME, DATASET])

""" LOADING DATASET """
print("\nLoading {} dataset...".format(DATASET), end=' ')
if DATASET == 'MNIST':
    (x_train, y_train), (x_test, y_test) = GCG_utils.get_MNIST(conv_reshape=True)
elif DATASET == 'EMNIST':
    (x_train, y_train), (x_test, y_test) = GCG_utils.get_EMNIST(conv_reshape=True)
else:
    raise ValueError("Don't know {} dataset. Program now quits.".format(DATASET))
print("done.")

# see memory footprint
GCG_utils.print_memory_footprint(x_train, y_train, x_test, y_test)

""" BUILDING THE MODELS """
print("Building the GAN models...", end=' ')
generator_model = DCGAN_build_generator(latent_dimension, name='{}_generator'.format(NAME))
discriminator_model = DCGAN_build_discriminator(img_shape, name='{}_discriminator'.format(NAME))
gan_model = tf.keras.models.Sequential([generator_model, discriminator_model], name='{}_gan'.format(NAME))

optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
discriminator_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)
discriminator_model.trainable = False

gan_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("done.", flush=True)

""" TRAIN THE MODEL IF IT DOES NOT EXIST """
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1000)
if not os.path.exists(PATH):
    print("Folder '{}'".format(PATH), "has not been found: training the model over", n_epochs, "epochs.")
    os.makedirs(PATH)
    os.makedirs(PATH + discriminator_model.name)
    os.makedirs(PATH + generator_model.name)
    os.makedirs(PATH + "train_images/")
    epoch_history_discriminator, epoch_history_gan = GCG_utils.train_DCGAN(gan_model, generator_model,
                                                                           discriminator_model, dataset,
                                                                           int(x_train.shape[0] / batch_size),
                                                                           latent_dimension, batch_size, n_epochs,
                                                                           path=PATH)

else:
    print("Folder '{}' has been found: loading model, no need to retrain.".format(NAME))
    generator_model = tf.keras.models.load_model(PATH + generator_model.name)
    discriminator_model = tf.keras.models.load_model(PATH + discriminator_model.name)
    gan_model = tf.keras.models.Sequential([generator_model, discriminator_model], name="{}_gan".format(NAME))
    with np.load(PATH + "training.npz") as load:
        epoch_history_discriminator = load['discr']  # loss, accuracy
        epoch_history_gan = load['gan']  # loss, accuracy

""" SEE RESULTS """
# plot losses
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_history_discriminator[:, 0])
plt.title("{} loss".format(discriminator_model.name))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(epoch_history_gan[:, 0])
plt.title("{} loss".format(gan_model.name))
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()

# plot accuracies
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_history_discriminator[:, 1])
plt.title("{} accuracy".format(discriminator_model.name))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.subplot(1, 2, 2)
plt.plot(epoch_history_gan[:, 1])
plt.title("{} accuracy".format(gan_model.name))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.show()

# plot images
for i in range(5):
    noise = tf.random.normal(shape=[25, latent_dimension])
    fake_images = generator_model(noise).numpy().reshape([25, 28, 28])
    for i in range(25):
        # define subplot
        plt.subplot(5, 5, 1 + i)
        plt.axis('off')
        plt.imshow(fake_images[i], cmap='gray_r')
    plt.show()  # see the results
    plt.close('all')
