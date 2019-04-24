from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

import tensorflow as tf
np.random.seed(111)


class MnistDataset(object):

    def __init__(self, x_train, y_train, x_test, y_test):
        self._train_x = x_train/255.0
        self._train_y = y_train
        self._test_x = x_test/255.0
        self._test_y = y_test

        self.train_dataset = None
        self.test_dataset = None
        self._batch = None
        self._test_batch = None
        self._val_x = None
        self._val_x_labels = []


    def prepare_data(self, config, binary=False, shuffle=True):
        '''
        config: given configurations
        '''
        self._test_batch = 0
        self._batch = 0
        
        if binary:
            self.test_x = (self._test_x[: int(len(self._test_x)/2), :]>0.5).astype(float)
            self.train_x = (self._train_x[: int(len(self._train_x)/2), :]>0.5).astype(float)
        else:
            self.test_x =  2. * self._test_x[: int(len(self._test_x)/2), :] - 1.
            self.train_x =  2. * self._train_x[: int(len(self._train_x)/2), :] - 1.

        self.test_x = np.reshape(self.test_x, [len(self.test_x), 28, 28])
        self.train_x = np.reshape(self.train_x, [len(self.train_x), 28, 28])


        # self.train_dataset = create_mnist_dataset(self._train_x, self._train_x, config.batch_size)
        # self.test_dataset = create_mnist_dataset(self._test_x, self._test_y, config.batch_size*3)

    def get_batch(self, batch_size, test = False, binary = False):
        '''
        returns the next train batch or test best
        increments batch counter
        '''
        if test:
            N = len(self.test_x) - batch_size
            index = (self._test_batch * batch_size) % N
            self._test_batch +=1
            return self.test_x[index:index+batch_size, :, :], self._test_y[index:index+batch_size]
        else:
            N = len(self._train_x) - batch_size
            index = (self._batch * batch_size) % N
            self._batch +=1
            return self.train_x[index:index+batch_size, :, :], self._train_y[index:index+batch_size]
            

    def get_validation_set(self, binary=False):
        '''
        '''        
        if binary:
            self._val_x = (self._test_x[int(len(self.test_x)):, :]>0.5).astype(float)
        else:
            self._val_x =  2*self._test_x[int(len(self.test_x)):, :] - 1.

        self._val_y = self._test_y[int(len(self.test_x)):] 
        self._val_x = np.reshape(self._val_x, [len(self._val_x), 28, 28])
        return self._val_x

    def get_digit_set(self, digit=1):
        '''
        '''
        self.get_validation_set()
        index = np.where(self._val_y == digit)
        return self._val_x[index]


    def reset(self, shuffle=True):
        '''
        after each epoch, shuffle the data
        '''
        if shuffle:
            s = np.arange(self._train_x.shape[0])
            np.random.shuffle(s)
            self._train_x = self._train_x[s, :, :]
            self._train_y = self._train_y[s]
        self._batch = 0
        self._test_batch = 0





