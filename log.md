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


02/08/2020 09:01
-----------------------------------------------------------------------------------------------------------------------
Now it starts being a little bit more succesfull. The DCGAN works appropriately well
but I need the big pc to train it more.

About the AEE, it seems this architecture achieves fairly good results:
>https://rubikscode.net/2019/01/21/generating-images-using-adversarial-autoencoders-and-python/

Anyway, time presses on. I will run both on the bigger pc as soon as I come home.
For now, it is time to write the report.
Also, I need to try another dataset, so I copy paste here the message:
>Dear all,
>
>in case you have choen the project on GANs, here you can find some datasets that can be used
>
>https://www.nist.gov/itl/products-and-services/emnist-dataset
>http://yann.lecun.com/exdb/mnist/
>https://davidstutz.de/fonts-a-synthetic-mnist-like-dataset-with-known-manifold/
>http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
>
>In some cases you need to adapt them to the size of MNIST letters (28x28).
>
>Regards,
>
>SM
>
>P.S. in case you want to try other things with GANs you can also check the datasets
>
>Fashion-MNIST https://github.com/zalandoresearch/fashion-mnist
>
>Pokemon https://www.kaggle.com/kvpratama/pokemon-images-dataset#101.png
>
>African fabrics https://www.kaggle.com/mikuns/african-fabric
>
>together with the more traditional ones (faces, dogs)