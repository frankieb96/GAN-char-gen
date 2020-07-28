import os
from tabulate import tabulate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import GCG_models
from tqdm import tqdm

tf.random.set_seed(1)
latent_dimension = 100