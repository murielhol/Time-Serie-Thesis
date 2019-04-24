import torch
import torch.nn as nn
import torch.nn.functional as F
import math;
import numpy as np
from torch.autograd import Variable

def gaussian_kld(left, right):
    """
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    mu_left, logvar_left = left; mu_right, logvar_right = right
    gauss_klds = 0.5 * (logvar_right - logvar_left +
                        (torch.exp(logvar_left) / torch.exp(logvar_right)) +
                        ((mu_left - mu_right)**2.0 / torch.exp(logvar_right)) - 1.0)
    return gauss_klds

def LogLikelihood(target, inputs, loss, data=None):

    if (loss == 'MSE'):
        mean = inputs[0];
        loss = (mean-target).pow(2)
        return -1.*loss

    elif (loss == 'L1'):
        mean = inputs[0];
        loss = torch.abs(mean-target)
        return -1.*loss
    
    elif (loss == 'Gaussian'):
        mean = inputs[0]; logvar = inputs[1];
        logvar = torch.clamp(logvar, -12., 12.)
        var = torch.exp(logvar);
        diff = target - mean;
        res = -torch.pow(diff,2)/(2 * torch.pow(var, 2));
        res = -0.5 * math.log(2 * math.pi) - logvar + res 
        return res;
    
    
