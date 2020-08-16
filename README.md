# GAN Character Generation
A machine learning, neural network based project that generates single characters with different
styles through adversarial learning.

**Packets requirements**: modules `numpy`, `tensorflow` (version 2 
and above, but should work with 1) and `keras`, then `scipy` for the .mat files loading, 
and `tabulate` and `tqdm` as utility packages. There are no shields, so all must be installed 
for the programs to work.

**Author**: Francesco Bianco

**Email**: francesco.bianco.5@studenti.unipd.it

### Quick intro

**Quick run**: from the work folder where all the .py files are stored, 
create a folder called 'temp_project' and put the 'matlab'
folder containing the EMNIST dataset inside it, so that a path of the kind
`./temp_project/matlab/emnist-letters.mat` (in Unix, or equivalently for Windows systems
`.\temp_project\matlab\emnist-letters.mat`) is valid.
Then launch the programs without command line arguments for default behaviour:
they will create folders inside 'temp_project' with 'E_' prefix for EMNIST-based network, 
and 'M_' for MNIST, followed by either 'DCGAN' or 'AAE'. 
The programs print a warning message that can be ignored.

At the end, inside the 'temp_project' folder, inside the subfolders are saved the models, 
training images and training history as a .npz file.

### Some notes

Since it may be desirable to run the programs in parallel, note that:
 - `GCG_emnist_DCGAN.py` and `GCG_mnist_DCGAN.py` are equivalent, and differs only 
   for the default behaviour (the former loads the EMNIST, the latter the MNIST),
   so there are two different files just for convenience.
   Same applies for `GCG_emnist_AAE.py` and `GCG_mnist_AAE.py`.
 - All nets are trained for a default of 15 epochs, and you need to manually change this
   value directly in the code for a different number.
 - At the end, they will show the results.
 - The programs first search if it exist a folder with the network name (e.g. if it 
   for this default, he may search for 'E_DCGAN'). If is found, then the training will
   not be repeated: the models are loaded and only the final results are shown. With
   this fashion, if any error arises before the models are saved, you need to manually 
   delete or change name to the folders, otherwise the programs will try to load the 
   models and crash. Same if you just want to repeat the training.
   
### Description 

#### File `GCG_utils.py`

Contains some useful functions to load the MNIST and the EMNIST, to visualize the memory 
occupation of the datasets, and to train the AAE and the DCGAN.

#### All the other files
The filenames are self-explanatory. 

All programs accept two optional arguments, but both must be specified, otherwise they
will be rolled back to default. These are 
 1. the folder name without blank spaces, and 
 2. the dataset to load, for which the only acceptable strings are, verbatim,
    "MNIST" or "EMNIST". Other strings will raise a ValueError. 

Examples: 
 - `python CGC_mnist_AAE.py new_folder EMNIST` will work,
 - `python CGC_mnist_AAE.py MNIST` will run with default parameters after a warning, 
 - `python CGC_mnist_AAE.py folder ABCD` will raise a ValueError and the program
   will quit saying `Don't know ABCD dataset. Program now quits.`.
 
 The code is the same for all classes:
  1. import the necessary modules,
  2. define the model creating function,
  3. define global variables and constants, among which the latent dimension, batch
     size and number of epochs,
  4. load the chosen dataset and visualize memory occupation; the MNIST is directly 
     integrated with keras, the EMNIST expects the matlab file version of the dataset,
  5. build and compile the keras models,
  6. train the model if it does not exist, or load it if instead it does,
  7. visualize plots of losses, accuracies, and some generated images.
