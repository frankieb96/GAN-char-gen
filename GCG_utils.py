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


def train_AAE():
    None
