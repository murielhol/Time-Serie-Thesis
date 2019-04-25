
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.finance.wavenet as wavenet
import models.finance.swavenet as swavenet

import torch

class Model(object):

    def __init__(self, config):

        self._config = config
        self.input_dim = len(self._config.features)
        self.embed_dim = config.num_hidden
        self.z_dim = config.num_hidden
        self.learning_rate = config.learning_rate

    def _build_model(self):
        '''
        '''
        self.gen = eval('swavenet').Model(self.input_dim, self.embed_dim, self.z_dim, 'L1', self._config.num_layers, self._config.num_stochastic_layers)
        nParams = sum([p.nelement() for p in self.gen.parameters()])
        print('* number of parameters: %d' % nParams)
        # !!!!!! gradient of generator is divided by 10 
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.learning_rate/10.0, eps=1e-5)


        self.dis = eval('wavenet').Model(self.input_dim, self.embed_dim, 'WSS', self._config.num_layers+self._config.num_stochastic_layers)
        nParams = sum([p.nelement() for p in self.dis.parameters()])
        print('* number of parameters: %d' % nParams)
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=self.learning_rate, eps=1e-5)
        
    