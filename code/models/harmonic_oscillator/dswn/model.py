
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.harmonic_oscillator.wavenet as wavenet
import models.harmonic_oscillator.swavenet as swavenet

import torch




class Model(object):

    def __init__(self, config):

        self._config = config
        self.input_dim = 1
        self.embed_dim = config.num_hidden
        self.z_dim = config.num_hidden
        self.learning_rate = config.learning_rate

    def _build_model(self):
        '''
        '''
        self.gen = eval('swavenet').Model(self.input_dim, self.embed_dim, self.z_dim, 'MSE', 4, 2)
        nParams = sum([p.nelement() for p in self.gen.parameters()])
        print('* number of parameters: %d' % nParams)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.learning_rate/10.0, eps=1e-5)


        self.dis = eval('wavenet').Model(self.input_dim, self.embed_dim, 'WSS', 6)
        nParams = sum([p.nelement() for p in self.dis.parameters()])
        print('* number of parameters: %d' % nParams)
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=self.learning_rate, eps=1e-5)
        
    