
import numpy as np 
import pandas as pd 
import sys
import os

from .dataset import Dataset
from .mnist_dataset import MnistDataset
from .wave_dataset import WaveDataset
from .ho_dataset import HoDataset

import tensorflow as tf

import matplotlib.pyplot as plt 

import random
import pickle 

np.random.seed(111)


def load_data(data_folder, config, skip = 1):

    if 'raw' in config.file_path:
        dataframe = pd.read_csv(data_folder + config.file_path, compression = 'gzip')
        dataframe = dataframe.reset_index()
        dataset = Dataset(dataframe)

    elif 'kaggle' in config.file_path:
        dataframe = pd.read_csv(data_folder + config.file_path)
        dataframe = dataframe.reset_index()
        dataset = Dataset(dataframe)

    elif 'coin' in config.file_path:
        dataframe = pd.read_csv(data_folder + config.file_path)
        dataframe = dataframe.reset_index()
        dataset = Dataset(dataframe)

    elif 'stock' in config.file_path:
        dataframe = pd.read_csv(data_folder + config.file_path)
        dataframe = dataframe.reset_index()
        dataset = Dataset(dataframe)
        
    elif 'mnist' in config.file_path:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        dataset = MnistDataset(x_train, y_train, x_test, y_test)


    elif 'ho' in config.file_path:
        train = 2*pickle.load( open( data_folder + "ho/ho_train.p", "rb" ) )-1
        test = 2*pickle.load( open( data_folder + "ho/ho_test.p", "rb" ) )-1
        val = 2*pickle.load( open( data_folder + "ho/ho_val.p", "rb" ) )-1
    
        dataset = HoDataset(np.expand_dims(train, axis=2),np.expand_dims(test, axis=2), np.expand_dims(val, axis=2))




    return dataset

