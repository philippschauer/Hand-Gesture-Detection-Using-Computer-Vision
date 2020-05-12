# List of all the imported packages, parameters and filenames
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import pickle
import os
import time
from numpy import unravel_index

from sklearn import svm, preprocessing, feature_selection
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.utils import class_weight
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers

# Do not show warnings from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters that are identical for all files, stored in a 
# well-arranged manner
Config={
    'num_classes': 5,
    'num_users': 9,
    'num_epochs': 10,
    'num_features': 13,

    'nn_hidden_1': 20,
    'nn_hidden_2': 10,
    'nn_batch_size': 10,

    'nn_model': 'model_nn.hdf5',
    'svm_model': 'model_svm.pkl',
    'bayes_model': 'model_bayes.pkl',
    'perceptron': 'model_perceptron.pkl',
    'neighbors': 'model_neighbors.pkl',
    'data_file': 'df.hdf5',

    'num_gamma': 10,
    'num_c': 10
}

# The End #
