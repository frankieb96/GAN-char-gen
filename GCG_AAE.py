import os
from tabulate import tabulate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import GCG_utils
from tqdm import tqdm


def AAE_build_encoder(img_shape=(28, 28), latent_dim=100, name='AAE_encoder'):
    input_layer = tf.keras.Input(img_shape)

    layers = tf.keras.layers.Flatten()(input_layer)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(latent_dim)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)

    model = tf.keras.Model(input_layer, layers, name=name)
    return model


def AAE_build_decoder(img_shape=(28, 28), latent_dim=100, name='AAE_decoder'):
    input_layer = tf.keras.Input(latent_dim)
    img_side = img_shape[0]

    layers = tf.keras.layers.Flatten()(input_layer)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(512)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Dense(img_side ** 2)(layers)
    layers = tf.keras.layers.LeakyReLU(alpha=0.2)(layers)
    layers = tf.keras.layers.Activation('tanh')(layers)
    layers = tf.keras.layers.Reshape((img_side, img_side))(layers)

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


""" GLOBAL VARIABLES AND CONSTANTS """
tf.random.set_seed(1)
latent_dimension = 10
img_shape = (28, 28)
batch_size = 32
n_epochs = 3
PATH = "temp_project/AAE/"

""" LOADING DATASET """
print("\nLoading MNIST dataset...", end=' ')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("done.")

""" NORMALIZATION """
print("Normalizing data...", end=' ')
x_train = (x_train / 255).astype(np.float32)
x_test = (x_test / 255).astype(np.float32)
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
print("Building the AAE model...", end=' ')
encoder_model = AAE_build_encoder(img_shape, latent_dimension)
decoder_model = AAE_build_decoder(img_shape, latent_dimension)
discriminator_model = AAE_build_discriminator(latent_dimension)
autoencoder_model = tf.keras.models.Sequential([encoder_model, decoder_model], name='AAE_autoencoder')
encoder_discriminator_model = tf.keras.models.Sequential([encoder_model, discriminator_model],
                                                         name='AAE_encoder_discriminator')

discriminator_model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)
discriminator_model.trainable = False

autoencoder_model.compile(
    optimizer='adam',
    loss='mse',
    loss_weights=[0.99]
)

encoder_discriminator_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    loss_weights=[0.01]
)
print("done.", flush=True)

""" TRAIN THE MODEL IF IT DOES NOT EXIST """
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1000)
if not os.path.exists("temp_project/AAE"):
    print("Folder '{}' has not been found: training the model over".format(PATH), n_epochs, "epochs.")
    os.makedirs(PATH)
    os.makedirs(PATH + discriminator_model.name)
    os.makedirs(PATH + encoder_model.name)
    os.makedirs(PATH + decoder_model.name)
    os.makedirs(PATH + "train_images/")
    GCG_utils.train_AAE(encoder_model, decoder_model, discriminator_model, autoencoder_model, encoder_discriminator_model,
                        dataset, path=PATH, total_batches=int(x_train.shape[0] / batch_size), n_epochs=n_epochs)
else:
    print("Folder '{}' has been found: loading model, no need to retrain.".format(PATH))
    discriminator_model = tf.keras.models.load_model(PATH + discriminator_model.name)
    encoder_model = tf.keras.models.load_model(PATH + encoder_model.name)
    decoder_model = tf.keras.models.load_model(PATH + decoder_model.name)
    autoencoder_model = tf.keras.models.Sequential([encoder_model, decoder_model], name='AAE_autoencoder')
    encoder_discriminator_model = tf.keras.models.Sequential([encoder_model, discriminator_model],
                                                             name='AAE_encoder_discriminator')

""" SEE RESULTS """
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
