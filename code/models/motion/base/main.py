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
        config = pickle.load( open( 'saved_models/'+config.model_name+'/config.p', "rb" ))
        config.validate = True
        config.file_path = file_path
        config.output_seq_length = output_len
        config.seed = seed
        print(config)
   

    t1 = time.time()
    data_folder =  os.path.abspath(os.path.abspath("../../../../"))+'/data/'
    dataset = load_data(data_folder, config)

    t2 = time.time()
    print('Finished loading the dataset: ' + str(t2-t1) +' sec \n')

    model = PricePredictor(config, dataset)

    if config.validate:
        # model._validate(steps = config.output_seq_length, epoch=400)
        model._make_figs()
    else:
        model._train()

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument('--file_path', type=str, default='motion', required=False, help="Name of the file that you want to train on")
    parser.add_argument('--model_name', type=str, default='motion_wn6', help='Unique name of the model')

    # Model params
    parser.add_argument('--input_seq_length', type=int, default=64, help='Length of the input sequence')
    parser.add_argument('--output_seq_length', type=int, default=10, help='Length of the output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in the model')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function generator')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--shuffle', default=True, help='If to shuffle the training set')
    parser.add_argument('--seed', type=int, default=111, help='Integer random seed')

    # Misc params
    parser.add_argument('--validate', default=True, help='If you want to validate the stored model')
    parser.add_argument('--backtest', default=False, help='')

    config = parser.parse_args()

    # Train the model
    main(config)
