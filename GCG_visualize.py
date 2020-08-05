import os
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

arr = np.max(dcgan_noise.numpy()[0])
arr1 = np.min(dcgan_noise.numpy()[0])

plt.subplot(2, 1, 1)
plt.imshow(aae_noise.numpy()[0], cmap='gray')
plt.yticks([])
plt.xticks([0, 10, 20, 27])
plt.title("AAE noise example")
plt.colorbar()
plt.subplot(2, 1, 2)
plt.imshow(noise_vec_to_matrix(dcgan_noise.numpy()[0], height=5), cmap='gray')
plt.yticks([])
plt.xticks([0, 20, 40, 60, 80, 99])
plt.title("DCGAN noise example")
plt.colorbar(orientation='horizontal', aspect=50)
plt.show()
print("done.")
