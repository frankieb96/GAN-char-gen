Francesco Bianco

Personal log

Created on 27/07/2020


26/07/2020 18:33
-----------------------------------------------------------------------------------------------------------------------
Created GAN-char-gen project.


28/07/2020 18:33
-----------------------------------------------------------------------------------------------------------------------
The simpleGAN works not so good, but it works nonetheless.
The DCGAN works far better just after 5 epochs.

Just for future project purposes, see this:

>https://towardsdatascience.com/a-wizards-guide-to-adversarial-autoencoders-part-1-autoencoder-d9a5f8795af4

Also, some useful links:
>EMNIST database [here](https://www.nist.gov/itl/products-and-services/emnist-dataset)
>with paper [here](https://arxiv.org/abs/1702.05373v1).
>Useful link for it:
> - https://towardsdatascience.com/how-to-load-matlab-mat-files-in-python-1f200e1287b5
> - https://www.kaggle.com/ashwani07/emnist-using-keras-cnn?scriptVersionId=10644687
> - https://github.com/NeilNie/EMNIST-Keras
> - https://github.com/christianversloot/extra_keras_datasets
> - https://www.kaggle.com/hojjatk/read-mnist-dataset


29/07/2020 08:47
-----------------------------------------------------------------------------------------------------------------------
Praise the Lord, and StackOverflow even more!
> https://stackoverflow.com/questions/51125969/loading-emnist-letters-dataset


01/08/2020 22:02
-----------------------------------------------------------------------------------------------------------------------
Still not working. 

See A. Geron, *Hands-on ML with Sklearn, Keras, and TF* (pdf), pagg. 772-773. Idea 
is to increase the latent space, but to add a regularizer l1 or l2 for sparsity.