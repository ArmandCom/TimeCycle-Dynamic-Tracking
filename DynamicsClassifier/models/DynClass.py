from collections import defaultdict
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .base_model import BaseModel
from models.networks.temporal_encoder_conv_AR import Estimator2D
from models.networks.selection import SelectionByAttention

from utils.hankel import gram_matrix
from utils import *

from torch.distributions import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO: trick! Switch from run to debug and vice versa --> hold Shift.

class DynClass(BaseModel):

  def __init__(self, opt):
    super(DynClass, self).__init__()

    self.delta = 1e-2

    self.is_train = opt.is_train
    self.image_size = opt.image_size[-1]
    self.T = opt.traj_length
    self.K = opt.k

    # Data parameters
    # self.__dict__.update(opt.__dict__)
    self.n_channels = opt.n_channels
    self.batch_size = opt.batch_size

    # Dimensions
    self.feat_latent_size = opt.feat_latent_size
    self.time_enc_size = opt.time_enc_size
    self.manifold_size = opt.manifold_size

    # Training parameters
    if opt.is_train:
      self.lr_init = opt.lr_init
      self.lr_decay = opt.lr_decay

    # Networks
    self.setup_networks()

    # Losses
    # self.loss_mse = nn.MSELoss() # reduction 'mean'
    # self.loss_l1 = nn.L1Loss(reduction='none')
    self.loss_dynamics = lambda S: torch.logdet(gram_matrix(S, delta=self.delta)).mean()

  def setup_networks(self):
    '''
    Networks for DDPAE.
    '''
    self.nets = {}

    # Time encoding network
    time_encoder_model = Estimator2D(
      code_length=self.T,
      fm_list=[4, 8, 16, 32],
      cpd_channels=self.feat_latent_size
    )
    self.time_encoder_model = nn.DataParallel(time_encoder_model.cuda())
    self.nets['time_encoder_model'] = self.time_encoder_model

    # Selection network
    selection_model = SelectionByAttention(in_channels= 2*self.feat_latent_size,
                                           indep_channels = self.T * self.K)
    self.selection_model = nn.DataParallel(selection_model.cuda())
    self.nets['selection_model'] = self.selection_model

    self.softmax = nn.Softmax(dim=1)

  def setup_training(self):
    '''
    Setup optimizers.
    '''
    if not self.is_train:
      return

    params = []
    # for name, net in self.nets.items():
    #   # if name != 'encoder_model': # Note: Impose decoder different optimizer than encoder
    #   params.append(net.parameters())
    #TODO: remember including all models in the optimizer
    self.optimizer = torch.optim.Adam( \
      [{'params': self.time_encoder_model.parameters(), 'lr': self.lr_init},
       {'params': self.selection_model.parameters(), 'lr': self.lr_init}
       ], betas=(0.9, 0.999))

    print('Parameters of time_encoder_model: ', self.time_encoder_model.parameters())

  def temporal_encoding(self, input):
    o = self.time_encoder_model(input)
    # Note: reverse? different networks?
    return o

  def selection_by_attention(self, input):
    o = self.selection_model(input)
    return o

  # def classify(self, latent):
  #   return self.decoder_model(latent)

  def train(self, input, step):

    input = Variable(input.cuda(), requires_grad=True)
    batch_size, n_chan, n_dim, traj_length = input.size()

    numel = batch_size * traj_length * n_dim
    loss_dict = {}

    '''Encode'''
    encoder_input = input[:, 0:2]
    scores = input[:, 2:3]
    #TODO: temporal encoding's convolution doesn't make sense if we use K selected.
    #TODO: if we allocate those k selected it would make sense (for dimensionality issues, we can split in x and y)
    t_enc = self.temporal_encoding(encoder_input)

    '''Selection by attention'''
    heat_map = self.selection_by_attention(t_enc)

    '''Resulting trajectory'''
    #Note: option 1: argmax !!Non-differentiable
    #TODO: mask, minimize entropy as in PPO, multiply (normalize by max?)
    # heat_map = torch.softmax(heat_map, dim=1)
    # heat_map = torch.sigmoid(heat_map)
    # val, idx = heat_map.max(1)
    # selected = torch.gather(encoder_input, dim=2, index=idx.view(batch_size, 2, 1, traj_length))

    #Note: option 2: attention + distributions + entropy
    heat_map = torch.sigmoid(heat_map)
    # heat_map = torch.softmax(heat_map, dim=1)
    entropy = 0
    distr = []
    samples = []
    for t in range(self.T):
      distr.append(Categorical(heat_map[...,t]))
      entropy += distr[t].entropy().mean()
      samples.append(distr[t].sample().view(batch_size, 2, 1)) #rsample not implemented for categorical

    idx = torch.stack(samples, dim=-1)
    heat_map.register_hook(print)
    selected = torch.gather(encoder_input, dim=2, index=idx)

    # # Apply softmax
    # z_dist = F.softmax(z_dist, dim=1)
    #
    # # Flatten out codes and distributions
    # z_d = z_d.view(len(z_d), -1).contiguous()
    # z_dist = z_dist.view(len(z_d), self.cpd_channels, -1).contiguous()
    #
    # # Log (regularized), pick the right ones
    # z_dist = torch.clamp(z_dist, self.eps, 1 - self.eps)
    # log_z_dist = torch.log(z_dist)
    # index = torch.clamp(torch.unsqueeze(z_d, dim=1) * self.cpd_channels, min=0,
    #                     max=(self.cpd_channels - 1)).long()
    # selected = torch.gather(log_z_dist, dim=1, index=index)
    # selected = torch.squeeze(selected, dim=1)

    # selected = None
    # if step%100 == 1:
    #   quick_represent(encoder_input[0, 0], selected.view(batch_size, 2, -1)[0,0])

    '''Losses'''
    #TODO: Normalize G?
    loss_dynamics = self.loss_dynamics(selected)
    loss_dict['Logdet_G'] = loss_dynamics.item()
    # print(loss_dynamics.item())

    '''Optimizer step'''
    loss = loss_dynamics
    loss_dict['Total_loss'] = loss.item()
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    return loss_dict

  def test(self, input, epoch=0, save_every=1):
    '''
    Return decoded output.
    '''
    res_dict = {}

    input = Variable(input.cuda())
    batch_size, n_chan, n_dim, traj_length = input.size()
    numel = batch_size * traj_length * n_dim
    # gt = torch.cat([input, output], dim=1)

    gt = input

    '''Time Encoding'''
    encoder_input = input[:, 0:2]
    scores = input[:, 2:3]
    t_enc = self.temporal_encoding(encoder_input)

    '''Selection by attention'''
    heat_map = self.selection_by_attention(t_enc)

    '''Resulting trajectory'''
    #Note: option 1: argmax !!Non-differentiable
    # heat_map = torch.softmax(heat_map, dim=1)
    # heat_map = torch.sigmoid(heat_map)
    # val, idx = heat_map.max(1)
    # selected = torch.gather(heat_map, dim=1, index=idx.unsqueeze(1))

    #Note: option 2: attention + distributions + entropy
    heat_map = torch.sigmoid(heat_map)
    # heat_map = torch.softmax(heat_map, dim=1)
    entropy = 0
    distr = []
    samples = []
    for t in range(self.T):
      distr.append(Categorical(heat_map[...,t]))
      entropy += distr[t].entropy().mean()
      samples.append(distr[t].sample().view(batch_size, 2, 1)) #rsample not implemented for categorical

    idx = torch.stack(samples, dim=-1)
    heat_map.register_hook(print)
    selected = torch.gather(encoder_input, dim=2, index=idx)

    '''Losses'''
    loss_dynamics = self.loss_dynamics(selected)
    res_dict['Logdet_G'] = loss_dynamics.item()

    '''Optimizer step'''
    loss = loss_dynamics
    res_dict['Total_loss'] = loss.item()

    '''Other Losses'''
    # if epoch % save_every:
    #   res_dict['Plot'] = self.save_visuals(gt, M, man, epoch)

    return res_dict

  def update_hyperparameters(self, epoch, n_epochs):

    lr_dict = super(DynClass, self).update_hyperparameters(epoch, n_epochs)

    return lr_dict

def quick_represent(self, all, selected):
  for i in range(self.K ):
    plt.plot(all[i].detach().cpu().numpy())
  plt.plot(selected.detach().cpu().numpy(), 'go--')
  plt.savefig('results/example_result')
  plt.close()


