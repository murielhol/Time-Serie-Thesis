from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import argparse
import time
import matplotlib.pyplot as plt
import pickle
import timeit


# add code folder to path
sys.path.insert(0, os.path.abspath("../../../"))
from utils.load_data import load_data
from price_predictor import PricePredictor


def main(config):

    
    if config.validate:
        output_len = config.output_seq_length
        config = pickle.load( open( 'saved_models/'+config.model_name+'/config.p', "rb" ))
        config.validate = True
        config.simulate = False
        config.output_seq_length = output_len
        print(config)


    t1 = time.time()
    data_folder =  os.path.abspath(os.path.abspath("../../../../"))+'/data/'
    dataset = load_data(data_folder, config)

    t2 = time.time()
    print('Finished loading the dataset: ' + str(t2-t1) +' sec \n')

    model = PricePredictor(config, dataset)
    if config.validate:
        model._make_figs(steps = config.output_seq_length, epoch=150)
        # model._validate(steps = config.output_seq_length, epoch=150)
    else:
        model._train()


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument('--file_path', type=str, default='ho_um', required=False, help="Name of the file that you want to train on")
    parser.add_argument('--model_name', type=str, default='SWN_ho_mse_fix_final', help='Unique name of the model')

    # Model params
    parser.add_argument('--input_seq_length', type=int, default=64, help='Length of an input sequence')
    parser.add_argument('--output_seq_length', type=int, default=50, help='Length of the output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--shuffle', default=True, help='If to shuffle the training set')
    parser.add_argument('--tensorboard', default=False, help='If to use tensorboard')

    # Misc params
    parser.add_argument('--validate', default=True, help='If only want to validate the stored model')


    config = parser.parse_args()

    # Train the model
    main(config)


