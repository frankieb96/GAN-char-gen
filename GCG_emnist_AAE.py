"""
Author: Francesco Bianco
Student number: 1234358
Email: francesco.bianco.5@studenti.unipd.it

Program file GCG_emnist_AAE.py
"""

import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import GCG_utils

""" ENCODER/DECODER/DISCRIMINATOR MODEL CREATOR FUNCTIONS """


def AAE_build_encoder(img_shape=(28, 28), latent_dimension=100, name='EMNIST_AAE_encoder'):
    encoder_input = tf.keras.layers.Input(img_shape)

    encoder_sequence = tf.keras.layers.Flatten()(encoder_input)
    encoder_sequence = tf.keras.layers.Dense(512)(encoder_sequence)
    encoder_sequence = tf.keras.layers.LeakyReLU(alpha=0.2)(encoder_sequence)
    encoder_sequence = tf.keras.layers.Dense(512)(encoder_sequence)
    encoder_sequence = tf.keras.layers.LeakyReLU(alpha=0.2)(encoder_sequence)
    latent_vector = tf.keras.layers.Dense(latent_dimension)(encoder_sequence)

    encoder_model = tf.keras.models.Model(encoder_input, latent_vector, name=name)
    return encoder_model


def AAE_build_decoder(img_shape=(28, 28), latent_dim=100, name='EMNIST_AAE_decoder'):
    input_layer = tf.keras.Input(latent_dim)

    layers = tf.keras.layers.Dense(512, input_dim=latent_dimension)(input_layer)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(np.prod(img_shape), activation='sigmoid')(layers)
    layers = tf.keras.layers.Reshape(img_shape)(layers)

    model = tf.keras.models.Model(input_layer, layers, name=name)
    return model


def AAE_build_discriminator(latent_dim=100, name='EMNIST_AAE_discriminator'):
    input_layer = tf.keras.Input(latent_dim)

    layers = tf.keras.layers.Flatten()(input_layer)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(256)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(1, activation='sigmoid')(layers)

    model = tf.keras.Model(input_layer, layers, name=name)
    return model


""" GLOBAL VARIABLES AND CONSTANTS """

DATA_TYPE = "emnist-letters"
tf.random.set_seed(1)
latent_dimension = 10
batch_size = 32
n_epochs = 15
if len(sys.argv) < 3:
    print("WARNING: not enough input params. Resorting to default params. Usage is 'AAE-name' 'dataset'",
          file=sys.stderr)
    NAME = "E_AAE"
    DATASET = "EMNIST"
else:
    NAME = sys.argv[1]
    DATASET = sys.argv[2]
PATH = "temp_project/{}/".format(NAME)
print("PARAMS are: ", [NAME, DATASET])

print("Loading data {}...".format(DATA_TYPE), end=' ')
(x_train, y_train), (x_test, y_test) = GCG_utils.get_EMNIST(conv_reshape=False)

# see memory footprint
GCG_utils.print_memory_footprint(x_train, y_train, x_test, y_test)

""" BUILDING THE MODELS """
print("Building the AAE model...", end=' ')
img_shape = (28, 28)
encoder_model = AAE_build_encoder(img_shape, latent_dimension, name='{}_encoder'.format(NAME))
decoder_model = AAE_build_decoder(img_shape, latent_dimension, name='{}_decoder'.format(NAME))
discriminator_model = AAE_build_discriminator(latent_dimension, name='{}_discriminator'.format(NAME))
autoencoder_model = tf.keras.models.Sequential([encoder_model, decoder_model], name='{}_autoencoder'.format(NAME))
encoder_discriminator_model = tf.keras.models.Sequential([encoder_model, discriminator_model],
                                                         name='{}_encoder_discriminator'.format(NAME))
optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
discriminator_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)
discriminator_model.trainable = False

autoencoder_model.compile(
    optimizer=optimizer,
    loss=['mse'],
    loss_weights=[0.99]
)

encoder_discriminator_model.compile(
    optimizer=optimizer,
    loss=['binary_crossentropy'],
    loss_weights=[0.01],
    metrics=['accuracy']
)
print("done.", flush=True)

""" TRAIN THE MODEL IF IT DOES NOT EXIST """
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1000)
if not os.path.exists(PATH):
    print("Folder '{}' has not been found: training the model over".format(PATH), n_epochs, "epochs.")
    os.makedirs(PATH)
    os.makedirs(PATH + discriminator_model.name)
    os.makedirs(PATH + encoder_model.name)
    os.makedirs(PATH + decoder_model.name)
    os.makedirs(PATH + "train_images/")
    epoch_history_autoenc, epoch_history_discriminator, epoch_history_encdiscr = GCG_utils.train_AAE(encoder_model,
                                                                                                     decoder_model,
                                                                                                     discriminator_model,
                                                                                                     autoencoder_model,
                                                                                                     encoder_discriminator_model,
                                                                                                     dataset, path=PATH,
                                                                                                     total_batches=int(
                                                                                                         x_train.shape[
                                                                                                             0] / batch_size),
                                                                                                     n_epochs=n_epochs)
else:
    print("Folder '{}' has been found: loading model, no need to retrain.".format(PATH))
    discriminator_model = tf.keras.models.load_model(PATH + discriminator_model.name)
    encoder_model = tf.keras.models.load_model(PATH + encoder_model.name)
    decoder_model = tf.keras.models.load_model(PATH + decoder_model.name)
    autoencoder_model = tf.keras.models.Sequential([encoder_model, decoder_model], name='{}_autoencoder'.format(NAME))
    encoder_discriminator_model = tf.keras.models.Sequential([encoder_model, discriminator_model],
                                                             name='{}_encoder_discriminator'.format(NAME))
    with np.load(PATH + "training.npz") as load:
        epoch_history_autoenc = load['autoenc']  # loss
        epoch_history_discriminator = load['discr']  # loss, accuracy
        epoch_history_encdiscr = load['encdiscr']  # loss, accuracy

""" SEE RESULTS """
# plot losses
plt.figure(figsize=(16, 5))
plt.subplot(1, 3, 1)
plt.plot(epoch_history_encdiscr[:, 0])
plt.title("{} loss".format(encoder_discriminator_model.name))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.subplot(1, 3, 2)
plt.plot(epoch_history_discriminator[:, 0])
plt.title("{} loss".format(discriminator_model.name))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.subplot(1, 3, 3)
plt.plot(epoch_history_autoenc)
plt.title("{} loss".format(autoencoder_model.name))
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()

# plot accuracies
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_history_encdiscr[:, 1])
plt.title("{} accuracy".format(encoder_discriminator_model.name))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.subplot(1, 2, 2)
plt.plot(epoch_history_discriminator[:, 1])
plt.title("{} accuracy".format(discriminator_model.name))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.show()

# plot images
for i in range(5):
    noise = tf.random.normal(shape=[25, img_shape[0], img_shape[1]])
    latent_real = autoencoder_model(noise).numpy()
    # plot images
    for i in range(25):
        # define subplot
        plt.subplot(5, 5, 1 + i)
        plt.axis('off')
        plt.imshow(latent_real[i].reshape((28, 28)), cmap='gray_r')
    plt.show()  # see the results
    plt.close('all')
