import os
from tabulate import tabulate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import GCG_utils
from tqdm import tqdm


def DCGAN_build_generator(latent_dimension=100, name='DCGAN_generator'):
    layer_input = tf.keras.Input(latent_dimension)

    layers = tf.keras.layers.Dense(7 * 7 * 128)(layer_input)
    layers = tf.keras.layers.Reshape([7, 7, 128])(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', activation='selu', kernel_initializer='lecun_normal')(layers)
    layers = tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', activation='tanh')(layers)

    # the output has shape (28, 28, 1)
    model = tf.keras.models.Model(layer_input, layers, name=name)
    return model


def DCGAN_build_discriminator(img_shape=(28, 28, 1), name='DCGAN_discriminator'):
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


tf.random.set_seed(1)
latent_dimension = 100
img_shape = (28, 28, 1)
batch_size = 32
n_epochs = 3

""" LOADING DATASET """
print("\nLoading MNIST dataset...", end=' ')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("done.")

""" NORMALIZATION """
print("Normalizing data...", end=' ')
x_train = (x_train / 255).astype(np.float32).reshape([x_train.shape[0], 28, 28, 1])
x_test = (x_test / 255).astype(np.float32).reshape([x_test.shape[0], 28, 28, 1])
print("done.")

# see memory footprint
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

""" BUILDING THE MODELS """
print("Building the GAN models...", end=' ')
generator_model = DCGAN_build_generator(latent_dimension)
discriminator_model = DCGAN_build_discriminator(img_shape)
gan_model = tf.keras.models.Sequential([generator_model, discriminator_model], name='DCGAN')

discriminator_model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)
discriminator_model.trainable = False

gan_model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)
print("done.", flush=True)

""" TRAIN THE MODEL IF IT DOES NOT EXIST """
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1000)
if not os.path.exists("temp_project/" + gan_model.name):
    print("Folder '{}'".format(gan_model.name), "has not been found: training the model over", n_epochs, "epochs.")
    os.makedirs("temp_project/" + gan_model.name)
    os.makedirs("temp_project/" + gan_model.name + "/" + discriminator_model.name)
    os.makedirs("temp_project/" + gan_model.name + "/" + generator_model.name)
    os.makedirs("temp_project/" + gan_model.name + "/train_images/")
    discriminator_history = GCG_utils.train_DCGAN(gan_model, generator_model, discriminator_model, dataset, int(x_train.shape[0] / batch_size), latent_dimension, batch_size, n_epochs)

    """
    # training
    for epoch in range(n_epochs):
        print("Epoch number", epoch + 1, "of", n_epochs, flush=True)
        for x_batch in tqdm(dataset, unit='batch', total=int(x_train.shape[0] / batch_size)):
            # train the discriminator
            noise = tf.random.normal(shape=[batch_size, latent_dimension])
            fake_images = generator_model(noise)
            x_tot = tf.concat([fake_images, x_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator_model.trainable = True
            discriminator_model.train_on_batch(x_tot, y1)
            discriminator_model.trainable = False

            # train the generator
            noise = tf.random.normal(shape=[batch_size, latent_dimension])
            y2 = tf.constant([[1.]] * batch_size)
            gan_model.train_on_batch(noise, y2)

        # save a sample at the end of each epoch
        noise = tf.random.normal(shape=[25, latent_dimension])
        fake_images = generator_model(noise).numpy().reshape([25, 28, 28])
        # plot images
        for i in range(25):
            # define subplot
            plt.subplot(5, 5, 1 + i)
            plt.axis('off')
            plt.imshow(fake_images[i], cmap='gray_r')
        if not os.path.isdir("temp_project/" + gan_model.name + "/train_images/"):
            os.makedirs("temp_project/" + gan_model.name + "/train_images/")
        plt.savefig("temp_project/" + gan_model.name + "/train_images/train_epoch_{}".format(epoch + 1))
        plt.close('all')
    print("Training complete. Saving the model...", end=' ')
    generator_model.save("temp_project\\" + gan_model.name + "\\" + generator_model.name)
    discriminator_model.save("temp_project\\" + gan_model.name + "\\" + discriminator_model.name)
    print("done.")
    """
else:
    print("Folder '{}'".format(gan_model.name), "has been found: loading model, no need to retrain.")
    generator_model = tf.keras.models.load_model("temp_project\\" + gan_model.name + "\\" + generator_model.name)
    discriminator_model = tf.keras.models.load_model("temp_project\\" + gan_model.name + "\\" + discriminator_model.name)
    gan_model = tf.keras.models.Sequential([generator_model, discriminator_model], name="SimpleGAN")

""" SEE RESULTS """
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