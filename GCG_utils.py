"""
Author: Francesco Bianco
Student number: 1234358
Email: francesco.bianco.5@studenti.unipd.it

Program file GCG_utils.py
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
from tabulate import tabulate


def get_MNIST(conv_reshape, normalize=True):
    """
    Loads and returns the MNIST dataset.

    :param conv_reshape: if True, reshapes the images to 28x28x1 for convolutional layers, otherwise are 28x28
    :param normalize: if True, already normalizes the images in [0, 1] as float32 numbers, else are 8 bit integers
    :return: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if conv_reshape:
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    if normalize:
        x_train = (x_train / 255).astype(np.float32)
        x_test = (x_test / 255).astype(np.float32)
    return (x_train, y_train), (x_test, y_test)


def get_EMNIST(conv_reshape, path='temp_project/matlab/', datatype='emnist-letters', normalize=True):
    """
    Loads and returns the EMNIST dataset.

    :param conv_reshape: if True, reshapes the images to 28x28x1 for convolutional layers, otherwise are 28x28
    :param path: a string to the path to the .mat files of the EMNIST dataset
    :param datatype: the .mat file name (without extension). Default is 'emnist-letters'
    :param normalize: if True, already normalizes the images in [0, 1] as float32 numbers, else are 8 bit integers
    :return:
    """
    mat = loadmat(path + "{}.mat".format(datatype))
    data = mat['dataset']
    x_train = data['train'][0, 0]['images'][0, 0]
    y_train = data['train'][0, 0]['labels'][0, 0]
    x_test = data['test'][0, 0]['images'][0, 0]
    y_test = data['test'][0, 0]['labels'][0, 0]

    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])

    if conv_reshape:
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1), order='F')
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1), order='F')
    else:
        x_train = x_train.reshape((x_train.shape[0], 28, 28), order='F')
        x_test = x_test.reshape((x_test.shape[0], 28, 28), order='F')
    if normalize:
        x_train = (x_train / 255).astype(np.float32)
        x_test = (x_test / 255).astype(np.float32)

    return (x_train, y_train), (x_test, y_test)


def print_memory_footprint(x_train, y_train, x_test, y_test):
    """
    Prints the memory occupations of the dataset. Since the numpy arrays contains built-in types, this is a real
    memory occupation.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    print("Memory footprint:\n")
    mb = lambda b: "{:.2f}".format(b / (1042 ** 2))
    headers = ["", "", "shape", "data type", "bytes", "Megabytes"]
    table = [["Training set", "x_train", x_train.shape, x_train.dtype, x_train.nbytes, mb(x_train.nbytes)],
             ["", "y_train", y_train.shape, y_train.dtype, y_train.nbytes, mb(y_train.nbytes)],
             [],
             ["Test set", "x_test", x_test.shape, x_test.dtype, x_test.nbytes, mb(x_test.nbytes)],
             ["", "y_test", y_test.shape, y_test.dtype, y_test.nbytes, mb(y_test.nbytes)]]
    print(tabulate(table, headers=headers))
    print("")


def train_DCGAN(gan_model, generator_model, discriminator_model, dataset, total_batches, latent_dimension=100,
                batch_size=32, n_epochs=10, path='temp_project/', verbose=True, save=True):
    """
    DCGAN training function.

    :param gan_model:
    :param generator_model:
    :param discriminator_model:
    :param dataset: the tf.Dataset object containing the input data
    :param total_batches: the total number of batches in the dataset (for the tqdm)
    :param latent_dimension: noise latent dimension of the generator. Default: 100
    :param batch_size: the minibatch size. Default: 32
    :param n_epochs: total number of training epochs. Default: 10
    :param path: the main path where to save the results. Default: './temp_project/'. From the specified path, results
    will be saved accordingly to the keras models 'name' parameter, plus a 'train_images' folder to store epochs ending
    images, and a 'training' file containing the histories vectors.
    :param verbose: unused.
    :param save: if True, it saves the results on secondary memory.
    :return: epoch_history_discriminator, epoch_history_gan, both as numpy arrays with loss in [:, 0] and accuracy in
    [:, 1]
    """
    # create an history object to save discriminator loss over the epochs
    # both have format ['loss', 'accuracy']
    epoch_history_discriminator = np.zeros([n_epochs, 2])
    epoch_history_gan = np.zeros([n_epochs, 2])
    epoch_index = 0
    for epoch in range(n_epochs):
        local_index = 0
        local_discr_history = np.zeros([total_batches, 2])
        local_gan_history = np.zeros([total_batches, 2])
        print("Epoch number", epoch + 1, "of", n_epochs, flush=True)
        for x_batch in tqdm(dataset, unit='batch', total=total_batches):
            # train the discriminator
            noise = tf.random.normal(shape=[batch_size, latent_dimension])
            fake_images = generator_model(noise)
            x_tot = tf.concat([fake_images, x_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator_model.trainable = True
            loss_acc = discriminator_model.train_on_batch(x_tot, y1)
            local_discr_history[local_index] = loss_acc
            discriminator_model.trainable = False

            # train the generator
            noise = tf.random.normal(shape=[batch_size, latent_dimension])
            y2 = tf.constant([[1.]] * batch_size)
            loss_acc = gan_model.train_on_batch(noise, y2)
            local_gan_history[local_index] = loss_acc

            local_index += 1

        epoch_history_discriminator[epoch_index] = np.array(
            [np.average(local_discr_history[:, 0]), np.average(local_discr_history[:, 1])])
        epoch_history_gan[epoch_index] = np.array(
            [np.average(local_gan_history[:, 0]), np.average(local_gan_history[:, 1])])
        epoch_index += 1

        if save:
            # save a sample at the end of each epoch
            noise = tf.random.normal(shape=[25, latent_dimension])
            fake_images = generator_model(noise).numpy().reshape([25, 28, 28])
            # plot images
            for local_index in range(25):
                # define subplot
                plt.subplot(5, 5, 1 + local_index)
                plt.axis('off')
                plt.imshow(fake_images[local_index], cmap='gray_r')
            plt.savefig(path + "train_images/train_epoch_{}".format(epoch + 1))
            plt.close('all')
    print("Training complete.")
    if save:
        print("Saving the models...", end=' ')
        generator_model.save(path + generator_model.name)
        discriminator_model.save(path + discriminator_model.name)
        np.savez(path + "training", discr=epoch_history_discriminator, gan=epoch_history_gan)
        print("done.")
    return epoch_history_discriminator, epoch_history_gan


def train_AAE(encoder_model, decoder_model, discriminator_model, autoencoder_model, encoder_discriminator_model,
              dataset, path, total_batches, img_shape=(28, 28), batch_size=32, n_epochs=10, verbose=True, save=True):
    """
    AAE training function.

    :param encoder_model:
    :param decoder_model:
    :param discriminator_model:
    :param autoencoder_model:
    :param encoder_discriminator_model:
    :param dataset: the tf.Dataset object containing the input data
    :param path: the main path where to save the results. From the specified path, results
    will be saved accordingly to the keras models 'name' parameter, plus a 'train_images' folder to store epochs ending
    images, and a 'training' file containing the histories vectors.
    :param total_batches: the total number of batches in the dataset (for the tqdm)
    :param img_shape: the input image shape. Default: (28, 28)
    :param batch_size: the minibatch size. Default: 32
    :param n_epochs: total number of training epochs. Default: 10
    :param verbose: unused.
    :param save: if True, it saves the results on secondary memory.
    :return: epoch_history_autoenc, epoch_history_discriminator, epoch_history_encdiscr, all as numpy arrays, the first
     with just the loss and the other two with loss in [:, 0] and accuracy in [:, 1]
    """
    # create an history object to save discriminator loss over the epochs
    # note: discriminator and encoder_discriminator have ['loss', 'accuracy'] metrics
    epoch_history_discriminator = np.zeros([n_epochs, 2])
    epoch_history_encdiscr = np.zeros([n_epochs, 2])
    epoch_history_autoenc = np.zeros(n_epochs)
    epoch_index = 0
    for epoch in range(n_epochs):
        local_index = 0
        discriminator_local_history = np.zeros([total_batches, 2])
        autoencoder_local_history = np.zeros(total_batches)
        encdiscr_local_history = np.zeros([total_batches, 2])

        print("Epoch number", epoch + 1, "of", n_epochs, flush=True)
        for x_batch in tqdm(dataset, unit='batch', total=total_batches):
            # train the discriminator
            noise = tf.random.normal(shape=[batch_size, img_shape[0], img_shape[1]])
            latent_real = encoder_model(noise)
            latent_fake = encoder_model(x_batch)
            x_tot = tf.concat([latent_real, latent_fake], axis=0)
            y1 = tf.constant([[1.]] * batch_size + [[0.]] * batch_size)
            discriminator_model.trainable = True
            loss = discriminator_model.train_on_batch(x_tot, y1)
            discriminator_local_history[local_index] = loss
            discriminator_model.trainable = False

            # train the autoencode reconstruction
            loss = autoencoder_model.train_on_batch(x_batch, x_batch)
            autoencoder_local_history[local_index] = loss

            # train the generator
            y2 = tf.constant([[1.]] * batch_size)
            loss = encoder_discriminator_model.train_on_batch(x_batch, y2)
            encdiscr_local_history[local_index] = loss

            local_index += 1

        # all are either in form ('loss', 'accuracy') or simply 'loss'
        epoch_history_discriminator[epoch_index] = np.array(
            [np.average(discriminator_local_history[:, 0]), np.average(discriminator_local_history[:, 1])])
        epoch_history_encdiscr[epoch_index] = np.array(
            [np.average(encdiscr_local_history[:, 0]), np.average(encdiscr_local_history[:, 1])])
        epoch_history_autoenc[epoch_index] = np.average(autoencoder_local_history)

        epoch_index += 1

        if save:
            # save a sample at the end of each epoch
            noise = tf.random.normal(shape=[25, img_shape[0], img_shape[1]])
            latent_real = autoencoder_model(noise).numpy()
            # plot images
            for i in range(25):
                # define subplot
                plt.subplot(5, 5, 1 + i)
                plt.axis('off')
                plt.imshow(latent_real[i].reshape(28, 28), cmap='gray_r')
            plt.savefig(path + "train_images/train_epoch_{}".format(epoch + 1))
            plt.close('all')

    print("Training complete.")
    if save:
        print("Saving the model...", end=' ')
        discriminator_model.save(path + discriminator_model.name)
        encoder_model.save(path + encoder_model.name)
        decoder_model.save(path + decoder_model.name)
        np.savez(path + "training", autoenc=epoch_history_autoenc, encdiscr=epoch_history_encdiscr,
                 discr=epoch_history_discriminator)
        print("done.")
    return epoch_history_autoenc, epoch_history_discriminator, epoch_history_encdiscr
