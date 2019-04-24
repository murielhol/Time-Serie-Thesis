
import os
import sys
import numpy as np
import pandas as pd
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
        config = pickle.load( open( 'saved_models/'+config.model_name+'/config.p', "rb" ))
        config.validate = True
        print(config)

    t1 = time.time()
    data_folder =  os.path.abspath(os.path.abspath("../../../"))+'/data/'
    dataset = load_data(data_folder, config)

    t2 = time.time()
    print('Finished loading the dataset: ' + str(t2-t1) +' sec \n')

    model = PricePredictor(config, dataset)
    if config.validate:
        model._validate(epoch=200)
        model._make_figs(epoch=200)
    elif config.tsne:
        model._tsne(epoch = 170)
    else:
        model._train()   


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument('--file_path', type=str, default='mnist', required=False, help="Name of the file that you want to train on")
    parser.add_argument('--model_name', type=str, default='wgan_mnist_team_final', help='Unique name of the model')

    # Model params
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding', type=int, default=128, help='Number of hidden units in the LSTM')

    # Training params
    parser.add_argument('--batch_size', type=int, default=24, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--critic_updates', type=float, default=1, help='Dropout')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--shuffle', default=True, help='If to shuffle the training set')
    parser.add_argument('--tensorboard', default=False, help='If to use tensorboard')

    # Misc params
    parser.add_argument('--print_every', type=int, default=500, help='How often to print training progress')
    parser.add_argument('--validate', default=True, help='If only want to validate the stored model')
    parser.add_argument('--tsne', default=False, help='If only want to validate the stored model')

    config = parser.parse_args()

    # Train the model
    main(config)


