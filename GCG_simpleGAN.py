"""
Program GCG_simpleGAN.py for the GAN character generation.

 Author: Francesco Bianco
 Mail: francesco.bianco.5@studenti.unipd.it
 Status: in development
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


""" LOAD DATASET """
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


""" VISUALIZATION """
for i in range(25):
    plt.subplot(5, 5, 1 + i)
    plt.axis('off')
    plt.title(y_train[i])
    plt.imshow(x_train[i], cmap='gray')
plt.show()


""" PREPROCESSING """
x_train = x_train / 255
x_test = x_test / 255


