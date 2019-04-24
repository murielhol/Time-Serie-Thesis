
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.finance.wavenet as wavenet

import torch


class Model(object):

    def __init__(self, config):

        self._config = config
        self.input_dim = len(config.features)
        self.embed_dim = config.num_hidden
        self.learning_rate = config.learning_rate

    def _build_model(self):
        
        self.gen = eval('wavenet').Model(self.input_dim, self.embed_dim, self._config.loss)
        nParams = sum([p.nelement() for p in self.gen.parameters()])
        print('* number of parameters: %d' % nParams)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.learning_rate, eps=1e-5)


                
    
