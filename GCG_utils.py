import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def train_DCGAN(gan_model, generator_model, discriminator_model, dataset, total_batches, latent_dimension=100, batch_size=32,
                n_epochs=10, path='temp_project/', verbose=True, save=True):
    # create an history object to save discriminator loss over the epochs
    epoch_history = np.zeros(n_epochs)
    epoch_index = 0
    for epoch in range(n_epochs):
        local_index = 0
        local_history = np.zeros(total_batches)
        print("Epoch number", epoch + 1, "of", n_epochs, flush=True)
        for x_batch in tqdm(dataset, unit='batch', total=total_batches):
            # train the discriminator
            noise = tf.random.normal(shape=[batch_size, latent_dimension])
            fake_images = generator_model(noise)
            x_tot = tf.concat([fake_images, x_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator_model.trainable = True
            discriminator_loss = discriminator_model.train_on_batch(x_tot, y1)
            local_history[local_index] = discriminator_loss
            local_index += 1
            discriminator_model.trainable = False

            # train the generator
            noise = tf.random.normal(shape=[batch_size, latent_dimension])
            y2 = tf.constant([[1.]] * batch_size)
            gan_model.train_on_batch(noise, y2)

        epoch_history[epoch_index] = np.average(local_history)
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
            plt.savefig(path + gan_model.name + "/train_images/train_epoch_{}".format(epoch + 1))
            plt.close('all')
    print("Training complete.")
    if save:
        print("Saving the models...", end= ' ')
        generator_model.save(path + gan_model.name + "/" + generator_model.name)
        discriminator_model.save(path + gan_model.name + "/" + discriminator_model.name)
        epoch_history.tofile(path + gan_model.name + "/discriminator_history")
        print("done.")
    return epoch_history


def train_AAE(encoder_model, decoder_model, discriminator_model, autoencoder_model, encoder_discriminator_model,
              dataset, path, total_batches, img_shape=(28,28), batch_size=32, n_epochs=10, verbose=True, save=True ):
    # create an history object to save discriminator loss over the epochs
    epoch_history_discriminator = np.zeros(n_epochs)
    epoch_history_autoencoder = np.zeros(n_epochs)
    epoch_index = 0
    for epoch in range(n_epochs):
        local_index = 0
        discriminator_local_history = np.zeros(total_batches)
        autoencoder_local_history = np.zeros(total_batches)
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
            local_index += 1

            # train the generator
            y2 = tf.constant([[1.]] * batch_size)
            encoder_discriminator_model.train_on_batch(x_batch, y2)

        epoch_history_discriminator[epoch_index] = np.average(discriminator_local_history)
        epoch_history_autoencoder[epoch_index] = np.average(autoencoder_local_history)
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
        epoch_history_autoencoder.tofile(path + "/autoencoder_history")
        epoch_history_discriminator.tofile(path + "/discriminator_history")
        print("done.")
    return epoch_history_autoencoder, epoch_history_discriminator

