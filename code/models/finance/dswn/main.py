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
        file_path = config.file_path
        seed = config.seed
        loss = config.loss
        l = config.num_layers
        sl = config.num_stochastic_layers
        config = pickle.load( open( 'saved_models/'+config.model_name+'/config.p', "rb" ))
        config.validate = True
        config.file_path = file_path
        config.output_seq_length = output_len
        config.seed = seed
        config.loss = loss
        config.num_layers = l
        config.num_stochastic_layers = sl
        config.backtest_target = 'close_btc'
        config.target = 'lr_btc'
        config.model_name = 'vaegan_mv_hour_'
        print(config)
   

    
    elif config.backtest:
        config = pickle.load( open( 'saved_models/'+config.model_name+'/config.p', "rb" ))
        config.backtest = True
        config.validate = False
        print(config)

    t1 = time.time()
    data_folder =  os.path.abspath(os.path.abspath("../../../../"))+'/data/'
    dataset = load_data(data_folder, config)

    t2 = time.time()
    print('Finished loading the dataset: ' + str(t2-t1) +' sec \n')
    model = PricePredictor(config, dataset)

    if config.validate:
        model._validate(steps = config.output_seq_length, epoch=500)
        # model._make_figs(steps = config.output_seq_length, epoch=200)
        # model._backtest(epoch=500)
    
    else:
        model._train()


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()
  
    parser.add_argument('--file_path', type=str, default='coins/hour/coins_mv_hour.csv', required=False, help="Name of the file that you want to train on")
    parser.add_argument('--model_name', type=str, default='vaegan_coins_hour_1', help='Unique name of the model')

    # Model params
    parser.add_argument('--input_seq_length', type=int, default=32, help='Length of an input sequence')
    parser.add_argument('--output_seq_length', type=int, default=5, help='Length of the output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers in the model')
    parser.add_argument('--num_stochastic_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--shuffle', default=True, help='If to shuffle the training set')
    parser.add_argument('--tensorboard', default=False, help='If to use tensorboard')
    parser.add_argument('--gp', type=float, default=0.1, help='gradient penalty')
    parser.add_argument('--critic_updates', type=int,  default=1, help='')
    parser.add_argument('--kld_epochs', type=int, default=100, help='')
    parser.add_argument('--kld_step', type=float, default=0.000001, help='')
    parser.add_argument('--kld_max', type=float, default=0.01, help='')
    parser.add_argument('--seed', default=False, help='random seed')
    parser.add_argument('--loss', type=str, default='Gaussian', help='loss function generator')


    # Misc params
    parser.add_argument('--validate', default=True, help='If want to validate the stored model')

    config = parser.parse_args()

    main(config)


