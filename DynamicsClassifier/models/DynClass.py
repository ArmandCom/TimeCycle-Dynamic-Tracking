from collections import defaultdict
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .base_model import BaseModel
# from models.networks.temporal_encoder_conv_AR import Estimator2D
from models.networks.selection_block_AR import TemporalEncoder
from models.networks.temporal_encoder_block_AR import TemporalEncoderSelector
from models.networks.selection import SelectionByAttention

from utils.hankel import gram_matrix
from utils import *
from utils.soft_argmax import SoftArgmax1D

from torch.distributions import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# TODO: trick! Switch from run to debug and vice versa --> hold Shift.

class DynClass(BaseModel):

  def __init__(self, opt):
    super(DynClass, self).__init__()

    self.delta = 1e-3

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

    lin_mode = 'ld'
    if lin_mode == 'ld':
        self.loss_dynamics = lambda S: torch.logdet(gram_matrix(S, delta=self.delta)).mean()
    elif lin_mode == 'tr':
        self.loss_dynamics = lambda S, flag='g': get_trace_K(M, flag).mean()
  def setup_networks(self):
    '''
    Networks for DDPAE.
    '''
    self.nets = {}

    # Time encoding network 1
    # time_encoder_model = Estimator2D(
    #   code_length=self.T,
    #   fm_list=[4, 8, 16, 32],
    #   cpd_channels=self.feat_latent_size
    # )

    # Time encoding network 2
    # time_encoder_model = TemporalEncoder(self.K, self.T, self.feat_latent_size, skip_connections=True)
    # self.time_encoder_model = nn.DataParallel(time_encoder_model.cuda())
    # self.nets['time_encoder_model'] = self.time_encoder_model

    # Time encoding network 3 - also selector
    time_encoder_model = TemporalEncoder(self.K, self.T, self.feat_latent_size, skip_connections=True)
    self.time_encoder_model = nn.DataParallel(time_encoder_model.cuda())
    self.nets['time_encoder_model'] = self.time_encoder_model

    # Selection network
    selection_model = SelectionByAttention(in_channels= self.feat_latent_size,
                                           indep_channels = self.T * self.K)
    # Note in_channels is 2*self_latent_size when we use the other AR Time Encoder.
    self.selection_model = nn.DataParallel(selection_model.cuda())
    self.nets['selection_model'] = self.selection_model

    # self.softmax = nn.Softmax(dim=1)
    # softargmax = SoftArgmax1D()
    # self.softargmax = nn.DataParallel(softargmax.cuda())

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

    batch_size, n_chan, n_dim, traj_length = input.size()

    # Note: Shuffle data in the K dimension --> Now done in data generation
    # for tr in range(self.T):
    #   k_idx_perm = torch.randperm(self.K)
    #   input[..., tr] = input[..., k_idx_perm, tr]

    # Note: unbias and normalize sequences
    # TODO: normalize single sequences in the source (before shuffling) --> would this be cheating?
    # seqs_norm = torch.norm(input[:,0:2].view(batch_size, 2, -1), dim=2).unsqueeze(-1).unsqueeze(-1)
    # input[:,0:2] = input[:,0:2]/seqs_norm
    seqs_mean = input[:,0:2].view(batch_size, 2, -1).mean(2).unsqueeze(-1).unsqueeze(-1)
    input[:,0:2] -= seqs_mean

    input = Variable(input.cuda(), requires_grad=True)


    numel = batch_size * traj_length * n_dim
    loss_dict = {}

    # Note: option 1: time encoding + selection different modules
    '''Encode'''
    encoder_input = input[:, 0:2]
    scores = input[:, 2:3]
    # t_enc = self.temporal_encoding(encoder_input)
    #
    # '''Selection by attention'''
    # heat_map = self.selection_by_attention(t_enc)

    # Note: option 2: selection by convolution + same-time-step correlations
    heat_map = self.temporal_encoding(encoder_input) # TODO: check gradients

    '''Resulting trajectory'''
    #Note: option 1: argmax !!Non-differentiable
    #TODO: mask, minimize entropy as in PPO, multiply (normalize by max?)
    # heat_map = torch.softmax(heat_map, dim=1)
    # heat_map = torch.sigmoid(heat_map)
    # val, idx = heat_map.max(1)
    # selected = torch.gather(encoder_input, dim=2, index=idx.view(batch_size, 2, 1, traj_length))

    #Note: option 2: attention + distributions + entropy + softargmax
    # # heat_map = torch.sigmoid(heat_map)
    # heat_map_softmax = torch.softmax(heat_map, dim=1)
    # entropy = 0
    # distr = []
    # indices = []
    # # samples = []
    # distr.append(Categorical(heat_map_softmax))
    # #TODO: this categorical is not right (computed over T)
    # for t in range(self.T):
    #   entropy += distr[t].entropy().mean()
    #   indices.append(self.softargmax(heat_map[...,t]))
    #   # samples.append(distr[t].sample().view(batch_size, 2, 1))
    # # idx = torch.stack(samples, dim=-1)
    # indices = torch.stack(indices, dim=-1)
    # idx = torch.round(indices).long()
    # selected = torch.gather(encoder_input.reshape(-1, n_dim, traj_length), dim=1, index=idx.unsqueeze(1))

    # Note: option 3: attention + distributions + entropy + softindices
    # heat_map = torch.sigmoid(heat_map)
    heat_map = torch.softmax(heat_map, dim=1) #TODO: Review option
    heat_map = heat_map / heat_map.max(1)[0].unsqueeze(1)
    entropy = 0
    distr = []
    soft_coord = []
    for t in range(self.T):
      distr.append(Categorical(heat_map[..., t]))
      entropy += distr[t].entropy().mean()
      soft_coord.append(torch.bmm(heat_map     [..., t].unsqueeze(-2),
                                  encoder_input[..., t].reshape(-1, n_dim, 1)).squeeze())
    soft_coord = torch.stack(soft_coord, dim=-1).unsqueeze(1)
    # selected = torch.round(soft_coord).long()
    # selected = torch.gather(encoder_input.reshape(-1, n_dim, traj_length), dim=1, index=indices.unsqueeze(1))

    # For representation:
    val, idx = heat_map.max(1)
    selected = torch.gather(encoder_input, dim=2, index=idx.view(batch_size, 2, 1, traj_length))

    # Note: option 4: directly output coordinate, minimize soft-euclidean distance with nearest neighbor.

    '''Losses'''
    #TODO: Try Trace of G
    loss_dynamics = self.loss_dynamics(soft_coord)
    loss_dict['Logdet_G'] = loss_dynamics.item()

    loss_entropy = entropy
    loss_dict['Entropy'] = loss_entropy.item()

    if step%100 == 1:
      quick_represent(encoder_input[0, 0], soft_coord.view(batch_size, 2, -1)[0,0], name='example_result_soft')
      quick_represent(encoder_input[0, 0], selected[0, 0, 0], name='example_result_hard')
      print('\n-loss_entropy: ', loss_entropy, '\nloss_dynamics: ', loss_dynamics)

    '''Optimizer step'''
    loss = loss_dynamics + 0.01 * loss_entropy
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

    # Note: option 1: time encoding + selection different modules
    '''Encode'''
    encoder_input = input[:, 0:2]
    scores = input[:, 2:3]
    # t_enc = self.temporal_encoding(encoder_input)
    #
    # '''Selection by attention'''
    # heat_map = self.selection_by_attention(t_enc)

    # Note: option 2: selection by convolution + same-time-step correlations
    heat_map = self.temporal_encoding(encoder_input)


    '''Resulting trajectory'''
    #Note: option 1: argmax !!Non-differentiable
    # heat_map = torch.softmax(heat_map, dim=1)
    # heat_map = torch.sigmoid(heat_map)
    # val, idx = heat_map.max(1)
    # selected = torch.gather(heat_map, dm=1, index=idx.unsqueeze(1))

    #Note: option 2: attention + distributions + entropy + softargmax
    # # heat_map = torch.sigmoid(heat_map)
    # heat_map_softmax = torch.softmax(heat_map, dim=1)
    # entropy = 0
    # distr = []
    # indices = []
    # # samples = []
    # distr.append(Categorical(heat_map_softmax))
    # #TODO: this categorical is not right (computed over T)
    # for t in range(self.T):
    #   entropy += distr[t].entropy().mean()
    #   indices.append(self.softargmax(heat_map[...,t]))
    #   # samples.append(distr[t].sample().view(batch_size, 2, 1))
    # # idx = torch.stack(samples, dim=-1)
    # indices = torch.stack(indices, dim=-1)
    # idx = torch.round(indices).long()
    # selected = torch.gather(encoder_input.reshape(-1, n_dim, traj_length), dim=1, index=idx.unsqueeze(1))

    # Note: option 3: attention + distributions + entropy + softindices
    # heat_map = torch.sigmoid(heat_map)
    heat_map = torch.softmax(heat_map, dim=1)
    heat_map = heat_map / heat_map.max(1)[0].unsqueeze(1)
    entropy = 0
    distr = []
    soft_coord = []
    for t in range(self.T):
      distr.append(Categorical(heat_map[..., t]))
      entropy += distr[t].entropy().mean()
      soft_coord.append(torch.bmm(heat_map[..., t].unsqueeze(-2),
                                  encoder_input[..., t].reshape(-1, n_dim, 1)).squeeze())
    soft_coord = torch.stack(soft_coord, dim=-1).unsqueeze(1)
    # selected = torch.round(soft_coord).long()
    # selected = torch.gather(encoder_input.reshape(-1, n_dim, traj_length), dim=1, index=indices.unsqueeze(1))

    # Note: option 4: directly output coordinate, minimize soft-euclidean distance with nearest neighbor.


    '''Losses'''
    loss_dynamics = self.loss_dynamics(soft_coord)
    res_dict['Logdet_G'] = loss_dynamics.item()

    loss_entropy = entropy
    res_dict['Entropy'] = loss_entropy.item()

    '''Optimizer step'''
    loss = loss_dynamics + 0.1*loss_entropy
    res_dict['Total_loss'] = loss.item()

    '''Other Losses'''
    # if epoch % save_every:
    #   res_dict['Plot'] = self.save_visuals(gt, M, man, epoch)

    return res_dict

  def update_hyperparameters(self, epoch, n_epochs):

    lr_dict = super(DynClass, self).update_hyperparameters(epoch, n_epochs)

    return lr_dict

def quick_represent(all, selected, name):
  for i in range(all.shape[0]):
    plt.plot(all[i].detach().cpu().numpy())
  plt.plot(selected.detach().cpu().numpy(), 'go--')
  plt.savefig('results/' + name)
  plt.close()


