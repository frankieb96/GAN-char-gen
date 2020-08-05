import math
import os
import sys

from et_xmlfile import xmlfile
from tabulate import tabulate
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def noise_vec_to_matrix(noise_vector, height=30, axis=0):
    out = np.zeros((height, noise_vector.shape[axis]), dtype=np.float32)
    for i in range(height):
        out[i] = noise_vector
    return out


def show_noise_example(aae_noise, dcgan_noise, show=True, save=False):
    plt.subplot(2, 1, 1)
    plt.imshow(aae_noise, cmap='gray')
    plt.yticks([])
    plt.xticks([0, 10, 20, 27])
    plt.title("AAE noise example")
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.imshow(noise_vec_to_matrix(dcgan_noise, height=5), cmap='gray')
    plt.yticks([])
    plt.xticks([0, 20, 40, 60, 80, 99])
    plt.title("DCGAN noise example")
    plt.colorbar(orientation='horizontal', aspect=50)
    if show:
        plt.show()
    if save:
        if not os.path.exists("temp_project/visualizer/"):
            os.makedirs("temp_project/visualizer/")
        plt.savefig("temp_project/visualizer/noise_example.png")
    plt.close('all')


""" SET GLOBAL VARIABLES AND CONSTANTS """
tf.random.set_seed(1)

aae_latent_dimension = 10
aae_img_shape = (28, 28)
aae_emnist_folder = "temp_project/EMNIST_AAE"
aae_mnist_folder = "temp_project/AAE"

dcgan_latent_dimension = 100
dcgan_img_shape = (28, 28, 1)
dcgan_emnist_folder = "temp_project/EMNIST_DCGAN"
dcgan_mnist_folder = "temp_project/DCGAN"

""" LOAD MNIST AND EMNIST-LETTERS """
EMNIST_DATA_TYPE = "emnist-letters"
EMNIST_PROJECT_FOLDER = "temp_project/EMINST_AAE/"

print("Loading and normalizing {} dataset...".format(EMNIST_DATA_TYPE.upper()), end=' ')
mat = loadmat("temp_project/matlab/{}.mat".format(EMNIST_DATA_TYPE))
data = mat['dataset']
emnist_train = data['train'][0, 0]['images'][0, 0]
emnist_label_train = data['train'][0, 0]['labels'][0, 0]
emnist_test = data['test'][0, 0]['images'][0, 0]
emnist_label_test = data['test'][0, 0]['labels'][0, 0]

emnist_train = (emnist_train.reshape((emnist_train.shape[0], 28, 28), order='F') / 255).astype(np.float32)
emnist_label_train = emnist_label_train.reshape(emnist_label_train.shape[0])
emnist_test = (emnist_test.reshape((emnist_test.shape[0], 28, 28), order='F') / 255).astype(np.float32)
emnist_label_test = emnist_label_test.reshape(emnist_label_test.shape[0])
print("done.")

# see memory footprint
print("{} memory footprint:\n".format(EMNIST_DATA_TYPE.upper()))
mb = lambda b: "{:.2f}".format(b / (1042 ** 2))
headers = ["{}".format(EMNIST_DATA_TYPE.upper()), "", "shape", "data type", "bytes", "Megabytes"]
table = [
    ["Training set", "x_train", emnist_train.shape, emnist_train.dtype, emnist_train.nbytes, mb(emnist_train.nbytes)],
    ["", "y_train", emnist_label_train.shape, emnist_label_train.dtype, emnist_label_train.nbytes,
     mb(emnist_label_train.nbytes)],
    [],
    ["Test set", "x_test", emnist_test.shape, emnist_test.dtype, emnist_test.nbytes, mb(emnist_test.nbytes)],
    ["", "y_test", emnist_label_test.shape, emnist_label_test.dtype, emnist_label_test.nbytes,
     mb(emnist_label_test.nbytes)]]
print(tabulate(table, headers=headers))
print("")

print("\nLoading and normalizing MNIST dataset...", end=' ')
(mnist_train, mnist_label_train), (mnist_test, mnist_label_test) = tf.keras.datasets.mnist.load_data()
mnist_train = (mnist_train / 255).astype(np.float32)
mnist_test = (mnist_test / 255).astype(np.float32)
print("done.")

# see memory footprint
print("MNIST memory footprint:\n")
mb = lambda b: "{:.2f}".format(b / (1042 ** 2))
headers = ["MNIST", "", "shape", "data type", "bytes", "Megabytes"]
table = [["Training set", "x_train", mnist_train.shape, mnist_train.dtype, mnist_train.nbytes, mb(mnist_train.nbytes)],
         ["", "y_train", mnist_label_train.shape, mnist_label_train.dtype, mnist_label_train.nbytes,
          mb(mnist_label_train.nbytes)],
         [],
         ["Test set", "x_test", mnist_test.shape, mnist_test.dtype, mnist_test.nbytes, mb(mnist_test.nbytes)],
         ["", "y_test", mnist_label_test.shape, mnist_label_test.dtype, mnist_label_test.nbytes,
          mb(mnist_label_test.nbytes)]]
print(tabulate(table, headers=headers))
print("")

""" CREATE NOISE """
print("Creating noise...", end=' ')
aae_noise = tf.random.normal(shape=[5, aae_img_shape[0], aae_img_shape[1]])
dcgan_noise = tf.random.normal(shape=[5, dcgan_latent_dimension])
show_noise_example(aae_noise.numpy()[0], dcgan_noise.numpy()[0], show=False, save=True)
print("done.")

""" LOAD MODELS """
print("Loading models...", end=' ')
aae_paths = [aae_mnist_folder, aae_emnist_folder, dcgan_emnist_folder, dcgan_mnist_folder]
models = {}
for path in aae_paths:
    if os.path.exists(path):
        last = path.split('/')[-1]
        # encoder
        if os.path.exists(path + "/{}_encoder".format(last)):
            encoder = tf.keras.models.load_model(path + "/{}_encoder".format(last))
            models[encoder.name] = encoder
        # decoder
        if os.path.exists(path + "/{}_decoder".format(last)):
            decoder = tf.keras.models.load_model(path + "/{}_decoder".format(last))
            models[decoder.name] = decoder
        # generator
        if os.path.exists(path + "/{}_generator".format(last)):
            generator = tf.keras.models.load_model(path + "/{}_generator".format(last))
            models[generator.name] = generator
        # discriminator
        if os.path.exists(path + "/{}_discriminator".format(last)):
            discriminator = tf.keras.models.load_model(path + "/{}_discriminator".format(last))
            models[discriminator.name] = discriminator
    else:
        print("WARNING: model folder '{}' does not exist. This model will not be loaded.".format(path), file=sys.stderr)
print("done. Loaded {} models:".format(len(models)), list(models.keys()))

aae = tf.keras.models.Sequential([models['AAE_encoder'], models['AAE_decoder']], name='MNIST_AAE')
out = aae(aae_noise).numpy()
print(out.shape)
x_ax = max([int(math.sqrt(out.shape[0])), 1])
y_ax = out.shape[0] - x_ax
plt.suptitle("Output of {}".format(aae.name))
for i in range(out.shape[0]):
    plt.subplot(x_ax, y_ax, 1 + i)
    plt.imshow(out[i], cmap='gray_r')
plt.show()
plt.close('all')

emnist_aae = tf.keras.Sequential([models['EMNIST_AAE_encoder'], models['EMNIST_AAE_decoder']], name='EMNIST_AAE')
out = emnist_aae(aae_noise).numpy()
print(out.shape)
x_ax = max([int(math.sqrt(out.shape[0])), 1])
y_ax = out.shape[0] - x_ax
plt.suptitle("Output of {}".format(emnist_aae.name))
for i in range(out.shape[0]):
    plt.subplot(x_ax, y_ax, 1 + i)
    plt.imshow(out[i], cmap='gray_r')
plt.show()
plt.close('all')

dcgan_generator = models['DCGAN_generator']
dcgan_discriminator = models['DCGAN_discriminator']
out = dcgan_generator(dcgan_noise).numpy()
out = out.reshape(out.shape[0], out.shape[1], out.shape[2])
print(out.shape)
x_ax = max([int(math.sqrt(out.shape[0])), 1])
y_ax = out.shape[0] - x_ax
plt.suptitle("Output of {}".format(dcgan_generator.name))
for i in range(out.shape[0]):
    plt.subplot(x_ax, y_ax, 1 + i)
    plt.imshow(out[i], cmap='gray_r')
plt.show()
plt.close('all')

emnist_dcgan_generator = models['EMNIST_DCGAN_generator']
emnist_dcgan_discriminator = models['EMNIST_DCGAN_discriminator']
out = emnist_dcgan_generator(dcgan_noise).numpy()
out = out.reshape(out.shape[0], out.shape[1], out.shape[2])
print(out.shape)
x_ax = max([int(math.sqrt(out.shape[0])), 1])
y_ax = out.shape[0] - x_ax
plt.suptitle("Output of {}".format(emnist_dcgan_generator.name))
for i in range(out.shape[0]):
    plt.subplot(x_ax, y_ax, 1 + i)
    plt.imshow(out[i], cmap='gray_r')
plt.show()
plt.close('all')

