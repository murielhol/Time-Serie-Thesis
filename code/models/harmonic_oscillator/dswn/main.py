from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

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
        config = pickle.load( open( 'saved_models/'+config.model_name+'/config.p', "rb" ))
        config.validate = True
        config.output_seq_length = output_len
        config.num_layers = 6
        print(config)
    
    elif config.backtest:
        config = pickle.load( open( 'saved_models/'+config.model_name+'/config.p', "rb" ))
        config.backtest = True
        print(config)

    t1 = time.time()
    data_folder =  os.path.abspath(os.path.abspath("../../../../"))+'/data/'
    dataset = load_data(data_folder, config)

    t2 = time.time()
    print('Finished loading the dataset: ' + str(t2-t1) +' sec \n')
    model = PricePredictor(config, dataset)

    if config.validate:
        model._validate(steps = config.output_seq_length, epoch=40)
        # model._make_figs(steps = config.output_seq_length, epoch=40)
    if config.backtest:
        model._backtest2( epoch=180)
    else:
        model._train()


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()
  
    parser.add_argument('--file_path', type=str, default='ho_um', required=False, help="Name of the file that you want to train on")
    parser.add_argument('--model_name', type=str, default='vaegan_ho_f', help='Unique name of the model')

    # Model params
    parser.add_argument('--input_seq_length', type=int, default=64, help='Length of an input sequence')
    parser.add_argument('--output_seq_length', type=int, default=50, help='Length of the output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of LSTM layers in the model')
    parser.add_argument('--num_stochastic_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=24, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--shuffle', default=True, help='If to shuffle the training set')
    parser.add_argument('--tensorboard', default=False, help='If to use tensorboard')
    parser.add_argument('--gp', default=False, help='gradient penalty')
    parser.add_argument('--critic_updates', default=False, help='')
    parser.add_argument('--kld_epochs', default=False, help='')
    parser.add_argument('--kld_step', default=False, help='')
    parser.add_argument('--kld_max', default=False, help='')

    # Misc params
    parser.add_argument('--validate', default=True, help='If want to validate the stored model')


    config = parser.parse_args()

    # Train the model
    main(config)


