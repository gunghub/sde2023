
from __future__ import print_function
import os, math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm_class
import torchvision
from PIL import Image
from copy import deepcopy
from model import Unet

class SDE(torch.nn.Module):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0, delta_t=0.001):
        super().__init__()
        self.beta_min = beta_min # beta 主要用来调控噪音的大小，隨着 t 的增大，噪音越大，
        self.beta_max = beta_max
        self.T = T   
        self.delta_t = delta_t

    @property
    def logvar_mean_T(self):
        logvar = torch.zeros(1)
        mean = torch.zeros(1)
        return logvar, mean

    def beta(self, timestamp_t):
        return self.beta_min + (self.beta_max-self.beta_min)*timestamp_t

    def f(self, timestamp_t, sample_batch):
        return - 0.5 * self.beta(timestamp_t) * sample_batch

    def g(self, timestamp_t, sample_batch): 
        beta_t = self.beta(timestamp_t)
        return torch.ones_like(sample_batch) * beta_t**0.5

    def add_noise(self, timestamp_t_batch, sample_x0_batch, return_noise):
        
        '''
        sample_batch shape: N, 2
        t: [0 , 1]
        See eq (32-33) of the paper
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        '''
        mean_weight=torch.exp(-0.25 * timestamp_t_batch**2 * (self.beta_max-self.beta_min) - 0.5 * timestamp_t_batch * self.beta_min)
        mean_mu = mean_weight * sample_x0_batch
        variance_batch=1. - torch.exp(-0.5 * timestamp_t_batch**2 * (self.beta_max-self.beta_min) - timestamp_t_batch * self.beta_min)
        standard_deviation_batch = variance_batch ** 0.5
        unit_noise_epsilon_batch = torch.randn_like(sample_x0_batch) # unit noise
        noisy_sample_xt_batch = unit_noise_epsilon_batch * standard_deviation_batch + mean_mu
        if not return_noise:
            return noisy_sample_xt_batch
        else:
          return noisy_sample_xt_batch, unit_noise_epsilon_batch, standard_deviation_batch, self.g(timestamp_t_batch, noisy_sample_xt_batch)


class ReverseSDE(torch.nn.Module):
    """
    The class represents an SDE (stochastic differential equation) with a given base SDE, drift function f, 
    and diffusion function g. The class also takes an inference SDE's drift function a and the time horizon T. 
    The purpose of this class is to invert the given base SDE with respect to the inference SDE.
    """
    def __init__(self, base_sde, drift_a, T):
        super().__init__()
        self.base_sde = base_sde
        self.model = drift_a
        self.T = T

    # Drift
    def mu(self, t, y, lmbd=0.):
        return (1. - 0.5 * lmbd) * self.base_sde.g(self.T-t, y) * self.model(y, self.T - t.squeeze()) - \
               self.base_sde.f(self.T - t, y)

    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T-t, y)

    @torch.enable_grad()
    def denoise_score_matching_loss(self, sample_x0_batch):
        """
        denoising score matching loss
        """
        # 默認
        timestamp_t_batch = torch.rand([sample_x0_batch.size(0), ] + [1 for _ in range(sample_x0_batch.ndim - 1)]).to(sample_x0_batch) * self.T
        noisy_sample_xt_batch, ground_truth_unit_noise_epsilon_batch, standard_deviation_batch, g = self.base_sde.add_noise(timestamp_t_batch, sample_x0_batch, return_noise=True)
        reverse_step_a = self.model(noisy_sample_xt_batch, timestamp_t_batch.squeeze())

        return ((reverse_step_a * standard_deviation_batch / g + ground_truth_unit_noise_epsilon_batch) ** 2).view(sample_x0_batch.size(0), -1).sum(1, keepdim=False) / 2

        # unit_noise_epsilon_batch × g / standard_devation_batch = - reverse_step
        # reverse_step 和 unit_noise 的方向是相反的。 



import functools

device = 'cuda'

def marginal_prob_std(t, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
  Returns The standard deviation.
  """    
  t = torch.tensor(t, device=device)
  # print(t.shape)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.
     returns the vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device=device)
  
sigma =  25.0 #@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)