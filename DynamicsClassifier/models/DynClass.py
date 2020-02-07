from collections import defaultdict
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .base_model import BaseModel
from models.networks.temporal_encoder_conv_AR import Estimator2D

from utils import *

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
    self.loss_mse = nn.MSELoss() # reduction 'mean'
    self.loss_l1 = nn.L1Loss(reduction='none')

  def setup_networks(self):
    '''
    Networks for DDPAE.
    '''
    self.nets = {}

    cpd_channels = 100 #TODO: choose
    # Time encoding
    time_encoder_model = Estimator2D(
            code_length=self.T,
            fm_list=[4, 4, 4, 4],
            cpd_channels=cpd_channels
        )
    self.time_encoder_model = nn.DataParallel(time_encoder_model.cuda())
    self.nets['time_encoder_model'] = self.time_encoder_model

    # Inverse Mapping
    # decoder_model = Decoder(self.n_frames_input, self.n_frames_output, self.n_channels,
    #                      self.image_size, self.feat_latent_size,
    #                      self.ngf, self.manifold_size)
    # self.decoder_model = nn.DataParallel(decoder_model.cuda())
    # self.nets['decoder_model'] = self.decoder_model

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
    self.optimizer = torch.optim.Adam(\
                     [{'params': self.time_encoder_model.parameters(), 'lr': self.lr_init}
                     ], betas=(0.9, 0.999))
    #TODO: check if parameters are optimizable
    print('Parameters of time_encoder_model: ', self.time_encoder_model.parameters())

  def temporal_encoding(self, input):

    o = self.time_encoder_model(input)
    # Note: reverse? different networks?

    return o

  # def classify(self, latent):
  #   return self.decoder_model(latent)

  def train(self, input):

    input = Variable(input.cuda(), requires_grad=False)
    batch_size, n_frames_input, n_dim = input.size()

    numel = batch_size * n_frames_input * n_dim
    loss_dict = {}

    '''Encode'''
    encoder_input = input
    x = self.temporal_encoding(encoder_input)

    '''Classify'''

    '''Losses'''
    # loss_dict['xxx'] = 0

    '''Optimizer step'''
    loss = x.sum()
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

    batch_size, _, n_dim = input.size()

    # gt = torch.cat([input, output], dim=1)
    gt = input

    '''Encode'''
    encoder_input = input
    x = self.encode(encoder_input)

    '''Classify'''

    '''Losses'''
    # if epoch % save_every:
    #   res_dict['Plot'] = self.save_visuals(gt, M, man, epoch)

    return res_dict

  def update_hyperparameters(self, epoch, n_epochs):

    lr_dict = super(DynClass, self).update_hyperparameters(epoch, n_epochs)

    return lr_dict
