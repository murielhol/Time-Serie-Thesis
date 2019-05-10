from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import argparse
import time


import pickle
import timeit


# add code folder to path
sys.path.insert(0, os.path.abspath("../../../"))
from utils.load_data import load_data
from price_predictor import PricePredictor

def main(config):

    
    if config.validate:
        output_len = config.output_seq_length
        file_path = config.file_path
        seed = config.seed
        loss = config.loss
        target = config.target
        config = pickle.load( open( 'saved_models/'+config.model_name+'/config.p', "rb" ))
        config.validate = True
        config.file_path = file_path
        config.output_seq_length = output_len
        config.seed = seed
        config.loss = loss
        config.backtest_target = 'close'
        config.target = 'NDX'
        print(config)
   

    t1 = time.time()
    data_folder =  os.path.abspath(os.path.abspath("../../../../"))+'/data/'
    dataset = load_data(data_folder, config)

    t2 = time.time()
    print('Finished loading the dataset: ' + str(t2-t1) +' sec \n')

    model = PricePredictor(config, dataset)

    if config.validate:
        # model._backtest(epoch=150)
        model._validate(steps = config.output_seq_length, epoch=150)
    else:
        model._train()

if __name__ == "__main__":

    # cols nasdaq
    cols = ['NDX', 'AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AKAM', 'ALXN', 'AMAT',
       'AMGN', 'AMZN', 'ATVI', 'AVGO', 'BBBY', 'BIDU', 'BIIB', 'CA', 'CELG',
       'CERN', 'CMCSA', 'COST', 'CSCO', 'CSX', 'CTRP', 'CTSH', 'DISCA', 'DISH',
       'DLTR', 'EA', 'EBAY', 'ESRX', 'EXPE', 'FAST', 'FB', 'FOX', 'FOXA',
       'GILD', 'GOOGL', 'INTC', 'JD', 'KHC', 'LBTYA', 'LBTYK', 'LRCX', 'MAR',
       'MAT', 'MCHP', 'MDLZ', 'MSFT', 'MU', 'MXIM', 'MYL', 'NCLH', 'NFLX',
       'NTAP', 'NVDA', 'NXPI', 'PAYX', 'PCAR', 'PYPL', 'QCOM', 'QVCA', 'ROST',
       'SBUX', 'SIRI', 'STX', 'SWKS', 'SYMC', 'TMUS', 'TRIP', 'TSCO', 'TSLA',
       'TXN', 'VIAB', 'VOD', 'VRTX', 'WBA', 'WDC', 'WFM', 'XLNX', 'YHOO']


    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument('--file_path', type=str, default='stock/nasdaq.csv', required=False, help="Name of the file that you want to train on")
    parser.add_argument('--features', type=list, default=cols, required=False, help="Names of the features to use as input")
    parser.add_argument('--target', type=str, default='NDX', required=False, help="Name of the features to use as target")
    parser.add_argument('--normalize', type=str, default='minmax', required=False, help="If to normalize")
    parser.add_argument('--test_size', type=float, default=0.1, help='fraction of the data, between 0.1 and 0.9')
    parser.add_argument('--model_name', type=str, default='nasdaq_wn_l2', help='Unique name of the model')
    parser.add_argument('--backtest_target', type=str, default='close', help='Unique name of the model')

    # Model params
    parser.add_argument('--input_seq_length', type=int, default=37, help='Length of the input sequence')
    parser.add_argument('--output_seq_length', type=int, default=1, help='Length of the output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers in the model')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function generator')


    # Training params
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--shuffle', default=True, help='If to shuffle the training set')
    parser.add_argument('--tensorboard', default=False, help='If to use tensorboard')
    parser.add_argument('--seed', type=int, default=111, help='Integer random seed')

    # Misc params
    parser.add_argument('--validate', default=True, help='If you want to validate the stored model')
    parser.add_argument('--backtest', default=False, help='')

    config = parser.parse_args()

    # Train the model
    main(config)
