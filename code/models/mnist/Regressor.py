import torch
import torch.nn as nn
import torch.nn.functional as F

class Regressor(nn.Module):
    def __init__(self, Loss, hid, m):
        super(Regressor, self).__init__()
        self.loss_function = Loss;
        self.m = m
        
        if (self.loss_function == 'MSE'):
            self.mean = nn.Linear(hid, self.m)
            self.tanh = nn.Tanh()

        elif (self.loss_function == 'Gaussian'):
            
            self.mean = nn.Linear(hid, self.m);
            self.var = nn.Linear(hid, self.m);
            self.tanh = nn.Tanh();

        elif (self.loss_function == 'WSS'):
            self.mean = nn.Linear(hid, self.m)

    def forward(self, X):
        
        batch_size, m = X.size(0), self.m;
        
        if (self.loss_function == 'MSE'):
            mean = self.tanh(self.mean(X))
            return [mean]
        
        elif (self.loss_function == 'Gaussian'):
            return [self.tanh(self.mean(X)), self.var(X)];
        
        elif (self.loss_function == 'WSS'):
            return self.mean(X);
        
 
