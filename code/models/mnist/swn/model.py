

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd 

import models.mnist.swavenet as swavenet

import torch

import timeit




class Model(object):

    def __init__(self, config):

        self._config = config
        self.input_dim = 28
        self.embed_dim = config.embedding
        self.z_dim = config.num_hidden
        self.learning_rate = config.learning_rate

    def _build_model(self):
        '''
        '''
        self.net = eval('swavenet').Model(self.input_dim, self.embed_dim, self.z_dim, 3, 1)
        nParams = sum([p.nelement() for p in self.net.parameters()])
        print('* number of parameters: %d' % nParams)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, eps=1e-5)
        
    