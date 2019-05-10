
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.motion.swavenet as swavenet

import torch


class Model(object):

    def __init__(self, config):

        self._config = config
        self.input_dim = 54
        self.embed_dim = config.num_hidden
        self.z_dim = config.num_hidden
        self.learning_rate = config.learning_rate
        self.loss = self._config.loss
        self.layers = self._config.num_layers
        self.stoch_layers = self._config.num_stochastic_layers

    def _build_model(self):
        
        self.gen = eval('swavenet').Model(self.input_dim, self.embed_dim, self.z_dim , self.loss, self.layers, self.stoch_layers)
        nParams = sum([p.nelement() for p in self.gen.parameters()])
        print('* number of parameters: %d' % nParams)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.learning_rate, eps=1e-5)


                
    
