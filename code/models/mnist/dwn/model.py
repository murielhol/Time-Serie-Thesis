
import numpy as np
import tensorflow as tf
import pandas as pd 

import models.mnist.wavenet as wavenet
import torch

np.random.seed(111)
tf.set_random_seed(111)

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
        self.gen = eval('wavenet').Model(self.input_dim, self.embed_dim, 'MSE', 4,   batch_norm=False, final_out_dim = 28)
        nParams = sum([p.nelement() for p in self.gen.parameters()])
        print('* number of parameters: %d' % nParams)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.learning_rate, eps=1e-5)

        self.dis = eval('wavenet').Model(self.input_dim, self.embed_dim, 'WSS', 4, batch_norm=False, final_out_dim = 28)
        nParams = sum([p.nelement() for p in self.dis.parameters()])
        print('* number of parameters: %d' % nParams)
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=self.learning_rate, eps=1e-5)


        
    