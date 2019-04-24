
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

np.random.seed(111)

class WaveDataset(object):

    '''
    TODOs:

    '''

    def __init__(self, data):
        self._data = data
        self._train = None
        self._test = None
        self._train_x = None
        self._train_y = None
        self._test_x = None
        self._test_x = None
        self._batch = None
        self._test_batch = None
        self._price_index = None

    def prepare_data(self, config, skip = 1, shuffle = True):
        '''
        config: given configurations
        shuffle: if train data is shuffled or not

        prepares the data for experiment by:
            - making sequences
            - initializing batch index 
        '''

        # splits in test and train
        train_size = int((1-config.test_size)*len(self._data))
        self.train_size = train_size
        data = self._data.copy()
        self._train_x, self._train_y = data[:train_size, :config.input_seq_length, :], data[:train_size, 1:config.input_seq_length+config.output_seq_length, :]
        self._test_x, self._test_y = data[train_size:, :config.input_seq_length, :], data[train_size:, 1:config.input_seq_length+config.output_seq_length, :]
        # initialize index of the current batch
        self._batch = 0
        self._test_batch = 0
        self._price_index = 0
        
    def get_batch(self, batch_size, test = False):
        '''
        returns the next train batch or test best
        increments batch counter
        '''
        if test:
            N = len(self._test_y) - batch_size
            # test set is smaller then train set so need a modula to restart the loop
            index = self._test_batch * batch_size % N
            self._test_batch +=1
            return self._test_x[index:index+batch_size, :, :], self._test_y[index:index+batch_size, :, :]
        else:
            index = self._batch * batch_size
            self._batch +=1
            return self._train_x[index:index+batch_size, :, :], self._train_y[index:index+batch_size, :, :]

    def reset(self, shuffle=True):
        '''
        after each epoch, shuffle the data
        '''
        if shuffle:
            s = np.arange(self._train_x.shape[0])
            np.random.shuffle(s)
            self._train_x = self._train_x[s, :, :]
            self._train_y = self._train_y[s, :, :]
            print('shuffling...')
        self._batch = 0


    def get_validation_set(self, input_seq_length, output_seq_length, skip= 1):
        '''
        The validation set is not shuffled so that temporal info is preserved
        '''
        return self._test_x, self._test_y, np.arange(len(self._train_y))


    