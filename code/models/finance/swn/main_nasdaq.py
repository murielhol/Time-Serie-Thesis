from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import time
import pickle

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
        # model._validate(steps = config.output_seq_length, epoch=200)
        model._backtest(epoch=200)
        # model._make_figs(epoch=300, steps = config.output_seq_length)

        
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
    parser.add_argument('--model_name', type=str, default='nasdaq_swn_mse_100', help='Unique name of the model')
    parser.add_argument('--backtest_target', type=str, default='close', help='Unique name of the model')

    # Model params
    parser.add_argument('--input_seq_length', type=int, default=32, help='Length of an input sequence')
    parser.add_argument('--output_seq_length', type=int, default=1, help='Length of the output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the model')
    parser.add_argument('--num_stochastic_layers', type=int, default=2, help='Number of stochastic layers in the model')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function generator')


    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--kld_step', type=float, default=0.0000001, help='Learning rate')
    parser.add_argument('--kld_max', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--kld_epochs', type=int, default=100, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--shuffle', default=True, help='If to shuffle the training set')
    parser.add_argument('--tensorboard', default=False, help='If to use tensorboard')
    parser.add_argument('--seed', default=False, help='random seed')

    # Misc params
    parser.add_argument('--validate', default=True, help='If only want to validate the stored model')


    config = parser.parse_args()

    # Train the model
    main(config)


