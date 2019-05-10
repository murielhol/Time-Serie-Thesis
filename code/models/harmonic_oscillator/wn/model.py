import models.harmonic_oscillator.wavenet as wavenet
import torch

class Model(object):

    def __init__(self, config):

        self._config = config
        self.input_dim = 1
        self.embed_dim = config.num_hidden
        self.z_dim = config.num_hidden
        self.learning_rate = config.learning_rate
        self.loss = config.loss
        self.layers = self._config.num_layers

    def _build_model(self):
        '''
        '''
        self.gen = eval('wavenet').Model(self.input_dim, self.embed_dim, self.loss, self.layers, batch_norm=False)
        nParams = sum([p.nelement() for p in self.gen.parameters()])
        print('* number of parameters: %d' % nParams)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.learning_rate, eps=1e-5)


                
    
