from scipy.io import loadmat
import os
from tabulate import tabulate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import GCG_utils
from tqdm import tqdm

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

batch_size = 32
n_epochs = 3
tf.random.set_seed(1)
latent_dimension = 100
img_shape = (28, 28, 1)
DATA_TYPE = "emnist-letters"
PATH = "temp_project/"

print("Loading data {}...".format(DATA_TYPE), end=' ')
mat = loadmat("temp_project/matlab/{}.mat".format(DATA_TYPE))
data = mat['dataset']
x_train = data['train'][0, 0]['images'][0, 0]
y_train = data['train'][0, 0]['labels'][0, 0]
x_test = data['test'][0, 0]['images'][0, 0]
y_test = data['test'][0, 0]['labels'][0, 0]

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1), order='F')
y_train = y_train.reshape(y_train.shape[0])
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1), order='F')
y_test = y_test.reshape(y_test.shape[0])
print("done.")

""" NORMALIZATION """
print("Normalizing data...", end=' ')
x_train = (x_train / 255).astype(np.float32)
x_test = (x_test / 255).astype(np.float32)
print("done.")

# see memory footprint
print("Memory footprint:")
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
dcgan_model = tf.keras.models.Sequential([generator_model, discriminator_model], name='EMNIST_DCGAN')

discriminator_model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)
discriminator_model.trainable = False

dcgan_model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)
print("done.", flush=True)

""" TRAIN THE MODEL IF IT DOES NOT EXIST """
end_epoch_noise = tf.random.normal(shape=[25, latent_dimension])
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1000)
if not os.path.exists("temp_project/" + dcgan_model.name):
    print("Folder '{}'".format(dcgan_model.name), "has not been found: training the model over", n_epochs, "epochs.")
    os.makedirs(PATH + dcgan_model.name + "/" + discriminator_model.name)
    os.makedirs(PATH + dcgan_model.name + "/" + generator_model.name)
    os.makedirs(PATH + dcgan_model.name + "/train_images/")
    GCG_utils.train_DCGAN(dcgan_model, generator_model, discriminator_model, dataset,
                          int(x_train.shape[0] / batch_size), latent_dimension, batch_size, n_epochs)
else:
    print("Folder '{}'".format(dcgan_model.name), "has been found: loading model, no need to retrain.")
    generator_model = tf.keras.models.load_model(PATH + generator_model.name)
    discriminator_model = tf.keras.models.load_model(PATH + discriminator_model.name)
    dcgan_model = tf.keras.models.Sequential([generator_model, discriminator_model], name="EMNIST_DCGAN")

""" SEE RESULTS """
# plot images
for i in range(5):
    noise = tf.random.normal(shape=[25, latent_dimension])
    fake_images = generator_model(noise).numpy()
    for i in range(25):
        # define subplot
        plt.subplot(5, 5, 1 + i)
        plt.axis('off')
        plt.imshow(fake_images[i].reshape(28, 28), cmap='gray_r')
    plt.show()  # see the results
    plt.close('all')
