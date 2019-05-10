
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

np.random.seed(111)

class Dataset(object):

    '''
    TODOs:

    '''

    def __init__(self, dataframe):
        self._df = dataframe
        self._train = None
        self._test = None
        self._train_x = None
        self._train_y = None
        self._test_x = None
        self._test_x = None
        self._batch = None
        self._test_batch = None
        self._price_index = None
        self.val_series = None 
        self.val_target = None
        self.val_ma_target = None
      

    def prepare_data(self, config, skip = 1, shuffle = True, skip_test = 1):
        '''
        config: given configurations
        shuffle: if train data is shuffled or not 

        prepares the data for experiment by:
            - normalizing (optional)
            - making sequences
            - initializing batch index 
        '''

        # splits in test and train
        train_size = int((1-config.test_size)*len(self._df))
        test_size = int((len(self._df) - train_size)/2)

        print('train_size: ', train_size, 'test_size: ', test_size)
        
        self.val_target = self._df[config.backtest_target].values[train_size+test_size:]

        self._df = self._df[list(config.features)]
        print('skip:', skip)
        print('skip_test: ', skip_test)

        if config.normalize == 'minmax':

            for col in config.features:
                print('Normalizing ', col)
                mi = np.min(self._df[col].copy().iloc[:train_size])
                ma = np.max(self._df[col].copy().iloc[:train_size])
                print('miama:', mi, ma)
                self._df[col] = 2*((self._df[col].values - mi)/(ma-mi)) - 1
                if col == config.target:
                    self.denorm_pars = [mi, ma]

        self.train = self._df.copy().iloc[:train_size, :]
        self.test = self._df.copy().iloc[train_size:train_size+test_size, :]
        self.val = self._df.copy().iloc[train_size+test_size:, :]

        self.val_series = np.expand_dims(self.val.copy(), 0)

        self._price_index = list(self._df.columns).index(config.target)
        del self._df

        # initialize index of the current batch
        self._batch = 0
        self._test_batch = 0
        self.train = self.train.as_matrix(columns=config.features)
        self.test = self.test.as_matrix(columns=config.features)
        self.val = self.val.as_matrix(columns=config.features)

    
        self.train_x, self.train_y = \
        self.make_sequences(np.array(self.train), config.input_seq_length, config.output_seq_length, skip = skip,  shuffle = config.shuffle)
        
        self.test_x, self.test_y = \
        self.make_sequences(np.array(self.test), config.input_seq_length, config.output_seq_length, skip = skip_test, shuffle = False)

        self.val_x, self.val_y = \
        self.make_sequences(np.array(self.val), config.input_seq_length, config.output_seq_length, skip = skip_test, shuffle = False)


        print('train size: ', np.shape(self.train_x))
        print('test size: ', np.shape(self.test_x))
        print('val size: ', np.shape(self.val_x))

    def get_batch(self, batch_size, test = False):
        '''
        returns the next train batch or test best
        increments batch counter
        '''
        
        if test:
            N = len(self.test_y) - batch_size
            # test set is smaller then train set so need a modula to restart the loop
            index = self._test_batch * batch_size % N
            self._test_batch +=1
    
            return self.test_x[index:index+batch_size, :, :], \
                    self.test_y[index:index+batch_size, :, :]

        else:
            index = self._batch * batch_size
            self._batch +=1
            return self.train_x[index:index+batch_size, :, :], \
                   self.train_y[index:index+batch_size, :, :]
        
    def reset(self, shuffle=True):
        '''
        after each epoch, shuffle the data
        '''
        if shuffle:
            s = np.arange(self.train_x.shape[0])
            np.random.shuffle(s)
            self.train_x = self.train_x[s, :, :]
            self.train_y = self.train_y[s, :, :]
            print('shuffling...')
        self._batch = 0


    def get_validation_set(self):
        '''
        The validation set is not shuffled so that temporal info is preserved
        Also needs to be denormalized to recapture the original trend
        '''
        # release from memory
        del self.train_x
        del self.train_y 
        del self.test_x
        del self.test_y 
        return self.val_x, self.val_y, self.val_target

    def get_backtest_set(self):
        '''
        '''
        # release from memory
        try:
            del self.train_x
            del self.train_y 
            del self.test_x
            del self.test_y 
        except AttributeError:
            pass
        return self.val_series, self.val_target



    def make_sequences(self, matrix, input_seq_length, output_seq_length, skip, shuffle=True):
        '''
        converts data from (number-of-timestamps x number-of-features) to
        (number-of-sequences x sequence-lenth x number-of-features) for the input
        and(number-of-sequences x sequence-lenth x 1) for the taget

        shuffles the sequences for training if required

        '''
        # make sequences
        data = []

        # create all possible sequences of length seq_len
        for index in range(0, len(matrix) - (input_seq_length+output_seq_length), skip): 
            sequence = matrix[index: index + (input_seq_length+output_seq_length)]
            data.append(sequence)

        data = np.array(data)

        if shuffle:
            s = np.arange(data.shape[0])
            np.random.shuffle(s)
            print('shuffling...')
            X = data[s, :input_seq_length, :]
            Y = data[s, 1:, :]
        else:
            X = data[:, :input_seq_length, :]
            Y = data[:, 1:, :]
        return X, Y

