
import models.harmonic_oscillator.wavenet as wavenet

import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(object):

    def __init__(self, config):

        self._config = config
        self.input_dim = 1
        self.embed_dim = config.embedding
        self.learning_rate = config.learning_rate

    def _build_model(self):
        '''
        '''
        self.gen = eval('wavenet').Model(self.input_dim, self.embed_dim, 'MSE', self._config.num_layers, batch_norm=False)
        nParams = sum([p.nelement() for p in self.gen.parameters()])
        print('* number of parameters: %d' % nParams)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.learning_rate, eps=1e-5)


                
    
