
import models.harmonic_oscillator.swavenet as swavenet

import torch


class Model(object):

    def __init__(self, config):

        self._config = config
        self.input_dim = 1
        self.embed_dim = config.num_hidden
        self.z_dim = config.num_hidden
        self.learning_rate = config.learning_rate
        self.loss = config.loss
        self.num_layers = config.num_layers
        self.num_stochastic_layers = config.num_stochastic_layers



    def _build_model(self):
        '''
        '''
        self.net = eval('swavenet').Model(self.input_dim, self.embed_dim, self.z_dim, self.loss, self.num_layers, self.num_stochastic_layers)
        nParams = sum([p.nelement() for p in self.net.parameters()])
        print('* number of parameters: %d' % nParams)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, eps=1e-5)
        
    