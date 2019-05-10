
import numpy as np 
import pandas as pd 
import sys
import os

from .dataset import Dataset
from .mnist_dataset import MnistDataset
from .ho_dataset import HoDataset
from .motion_dataset import MotionDataset

import tensorflow as tf

import matplotlib.pyplot as plt 

import random
import pickle 


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
    
        dataset = HoDataset(np.expand_dims(train, axis=2), np.expand_dims(test, axis=2), np.expand_dims(val, axis=2))

    elif 'motion' in config.file_path:
        actions, train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = pickle.load( open( data_folder + "motion/human_motion_data.p", "rb" ) ) 
        dataset = MotionDataset(actions, train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use)



    return dataset

