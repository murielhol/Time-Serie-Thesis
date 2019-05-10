import torch
import torch.nn as nn
import torch.nn.functional as F

class Regressor(nn.Module):
    def __init__(self, Loss, hid, m):
        super(Regressor, self).__init__()
        self.loss_function = Loss;
        self.m = m
        
        if (self.loss_function == 'MSE' or self.loss_function == 'L1'):
            self.mean = nn.Linear(hid, self.m)

      
        if (self.loss_function == 'Gaussian'):
            
            self.mean = nn.Linear(hid, self.m);
            self.var = nn.Linear(hid, self.m);
            
        
            
    def forward(self, X):
        
        batch_size, m = X.size(0), self.m;

        if (self.loss_function == 'MSE' or self.loss_function == 'L1'):
            mean = self.mean(X)
            return [mean]
        
        if (self.loss_function == 'Gaussian'):
            
            return [self.mean(X), self.var(X)];
            
       
 
