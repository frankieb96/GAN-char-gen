import os
from tabulate import tabulate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import GCG_models
from tqdm import tqdm

tf.random.set_seed(1)
latent_dimension = 100

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
img_shape = (28, 28)
latent_dimension = 10
encoder_model = GCG_models.AAE_build_encoder(img_shape, latent_dimension)
decoder_model = GCG_models.AAE_build_decoder(img_shape, latent_dimension)
discriminator_model = GCG_models.AAE_build_discriminator(latent_dimension)
autoencoder_model = tf.keras.models.Sequential([encoder_model, decoder_model], name='AAE_autoencoder')
encoder_discriminator_model = tf.keras.models.Sequential([encoder_model, discriminator_model], name='AAE_encoder_discriminator')

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
batch_size = 32
n_epochs = 10
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1000)
if not os.path.exists("temp_project/AAE"):
    print("Folder 'AAE' has not been found: training the model over", n_epochs, "epochs.")
    os.makedirs("temp_project/AAE")
    os.makedirs("temp_project/AAE" + "/" + discriminator_model.name)
    os.makedirs("temp_project/AAE" + "/" + encoder_model.name)
    os.makedirs("temp_project/AAE" + "/" + decoder_model.name)

    # training
    for epoch in range(n_epochs):
        print("Epoch number", epoch + 1, "of", n_epochs, flush=True)
        for x_batch in tqdm(dataset, unit='batch', total=int(x_train.shape[0] / batch_size)):

            # train the discriminator
            noise = tf.random.normal(shape=[batch_size, img_shape[0], img_shape[1]])
            latent_real = encoder_model(noise)
            latent_fake = encoder_model(x_batch)
            x_tot = tf.concat([latent_real, latent_fake], axis=0)
            y1 = tf.constant([[1.]] * batch_size + [[0.]] * batch_size)
            discriminator_model.trainable = True
            discriminator_model.train_on_batch(x_tot, y1)
            discriminator_model.trainable = False

            # train the autoencode reconstruction
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            autoencoder_model.train_on_batch(imgs, imgs)

            # train the generator
            y2 = tf.constant([[1.]] * batch_size)
            encoder_discriminator_model.train_on_batch(x_batch, y2)

        # save a sample at the end of each epoch
        noise = tf.random.normal(shape=[25, img_shape[0], img_shape[1]])
        latent_real = autoencoder_model(noise).numpy()
        # plot images
        for i in range(25):
            # define subplot
            plt.subplot(5, 5, 1 + i)
            plt.axis('off')
            plt.imshow(latent_real[i].reshape(28, 28), cmap='gray_r')
        if not os.path.isdir("temp_project/AAE/train_images/"):
            os.makedirs("temp_project/AAE/train_images/")
        plt.savefig("temp_project/AAE/train_images/train_epoch_{}".format(epoch + 1))
        plt.close('all')
    print("Training complete. Saving the model...", end=' ')
    discriminator_model.save("temp_project\\AAE\\" + discriminator_model.name)
    encoder_model.save("temp_project\\AAE\\" + encoder_model.name)
    decoder_model.save("temp_project\\AAE\\" + decoder_model.name)
    print("done.")
else:
    print("Folder 'AAN' has been found: loading model, no need to retrain.")
    discriminator_model = tf.keras.models.load_model("temp_project\\AAE\\" + discriminator_model.name)
    encoder_model = tf.keras.models.load_model("temp_project\\AAE\\" + encoder_model.name)
    decoder_model = tf.keras.models.load_model("temp_project\\AAE\\" + decoder_model.name)
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
