
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .base_model import BaseModel
# from models.networks.selection_AR_encoder_w_inicond import TemporalEncoder
# from models.networks.mapping import Encoder as TemporalEncoder
from models.networks.mapping_ini_cond import Encoder as TemporalEncoder

from utils.hankel import gram_matrix, JBLDLoss_rolling
from utils.utils import closest_sequence, distance_closest_sequence, weighted_distance_closest_sequence
from utils.utils import format_input
from utils import *
from torchvision.transforms import Normalize

from torch.distributions import *

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class DynClass(BaseModel):

  def __init__(self, opt):
    super(DynClass, self).__init__()

    self.delta = 1e-5
    self.weight_1nn = 15
    # self.weight_1nn = 5

    self.shape = 32
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
    self.t_enc_rnn_hidden_size = opt.t_enc_rnn_hidden_size
    self.trans_rnn_hidden_size = opt.trans_rnn_hidden_size

    self.ini_length = opt.ini_length

    self.manifold_size = 1

    # Training parameters
    if opt.is_train:
      self.lr_init = opt.lr_init
      self.lr_decay = opt.lr_decay

    # Networks
    self.setup_networks()

    # Losses
    self.loss_mse = nn.MSELoss() # reduction 'mean'
    self.loss_mse_keepdim = nn.MSELoss(reduction='none')
    # self.loss_l1 = nn.L1Loss(reduction='none')

    # Note: Whole sequence logdet
    self.loss_dynamics = lambda S: torch.logdet(gram_matrix(S, diff=True, delta=self.delta)).mean() #Note: [...,:-1], diff = False
    # Note: Rolling window logdet
    # self.loss_dynamics, self.subtraj_length = lambda S: torch.logdet(gram_matrix(S, diff=False, delta=self.delta)), self.T-1
    # Note: JBLD - TODO: change max to mean ?
    # self.loss_dynamics_jbld = lambda S: JBLDLoss_rolling(S, delta=self.delta, sz=7).max()
    self.loss_dynamics_jbld = lambda S: JBLDLoss_rolling(S, delta=self.delta, sz=7).mean()


    # Note: Mean distance to NNs
    # self.loss_1nn = lambda S, Sall: self.loss_mse(S, closest_sequence(S, Sall))
    # Note: Max distance to NN
    self.loss_1nn = lambda S, Sall: self.loss_mse_keepdim(S, closest_sequence(S, Sall)).max(-1)[0].mean()
    # self.loss_1nn = lambda S, Sall: (weighted_distance_closest_sequence(S, Sall)).max(-1)[0].mean()

  def setup_networks(self):
    '''
    Networks for DDPAE.
    '''
    self.nets = {}

    # Time encoding network 4 - Koopman approach

    time_encoder_model = TemporalEncoder(self.T, self.ini_length, n_channels=1,
                                         image_size = [1, self.shape, self.T], # or reversed?
                                         feat_latent_size=self.feat_latent_size,
                                         time_enc_size=self.time_enc_size,
                                         t_enc_rnn_hidden_size=self.t_enc_rnn_hidden_size,
                                         trans_rnn_hidden_size=self.trans_rnn_hidden_size,
                                         manifold_size=self.manifold_size)
    self.time_encoder_model = nn.DataParallel(time_encoder_model.cuda())
    self.nets['time_encoder_model'] = self.time_encoder_model

    # self.softmax = nn.Softmax(dim=1)
    # softargmax = SoftArgmax1D()
    # self.softargmax = nn.DataParallel(softargmax.cuda())

  def setup_training(self):
    '''
    Setup optimizers.
    '''
    if not self.is_train:
      return

    self.optimizer = torch.optim.Adam( \
      [{'params': self.time_encoder_model.parameters(), 'lr': self.lr_init}
       ], betas=(0.9, 0.999))

    print('Parameters of time_encoder_model: ', self.time_encoder_model.parameters())

  def temporal_encoding(self, input):
    o = self.time_encoder_model(input)
    return o

  # def classify(self, latent):
  #   return self.decoder_model(latent)

  def train(self, input, step):

    # Note: unbias and normalize sequences
    # seqs_norm = torch.norm(input[:,0:2].view(batch_size, 2, -1), dim=2).unsqueeze(-1).unsqueeze(-1)
    # input[:,0:2] = input[:,0:2]/seqs_norm
    # seqs_mean = input[:,0:2].view(batch_size, 2, -1).mean(2).unsqueeze(-1).unsqueeze(-1)
    # input[:,0:2] -= seqs_mean

    # TODO: mask forks - maybe in origin
    input_score, input_coord, input_reshaped = format_input(input, 0, self.shape, save_test=False)

    input = Variable(input_score.cuda(), requires_grad=True)
    input_reshaped = Variable(input_reshaped.cuda(), requires_grad=True)

    batch_size, n_chan, n_dim, traj_length = input_reshaped.size()
    numel = batch_size * traj_length * n_dim
    loss_dict = {}

    # Note: option 1: time encoding + selection different modules
    '''Encode'''
    encoder_input = (input , input_reshaped) # Note: we add input reshaped only in second version of model
    soft_coord = self.temporal_encoding(encoder_input)

    # Give ground truth to the first point.
    # soft_coord = torch.cat([input_reshaped[..., 0:1, 0:1],soft_coord[..., 1:]], dim=-1)

    #  Note: Option 1: loss_dynamics iteratively
    # loss_dynamics = torch.FloatTensor((-1e7,)).cuda().repeat(soft_coord.shape[0])
    # for t_init in range(self.T - self.subtraj_length + 1):
    #   loss_dynamics = torch.max(loss_dynamics, self.loss_dynamics(soft_coord.squeeze(-2)[..., t_init:t_init+self.subtraj_length]))
    # loss_dynamics = loss_dynamics.mean()

    #  Note: Option 2: loss_dynamics at once
    loss_dynamics = self.loss_dynamics(soft_coord.squeeze(-2))
    loss_dict['Logdet_G'] = loss_dynamics.item()

    loss_1nn = self.loss_1nn(soft_coord, input_reshaped)
    loss_dict['loss_1nn'] = loss_1nn.item()

    if step%25 == 1:
      selected = closest_sequence(soft_coord, input_reshaped)
      quick_represent(input_reshaped[0, 0], soft_coord.view(batch_size, -1)[0],
                      name='example_result_soft')
      # quick_represent(input_reshaped[0, 0], selected[0, 0, 0], name='example_result_hard')
      print('\n-loss_1nn: ', loss_1nn, '\nloss_dynamics: ', loss_dynamics)

    '''Optimizer step'''
    loss = loss_dynamics + self.weight_1nn * loss_1nn
    loss_dict['Total_loss'] = loss.item()
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    return loss_dict

  def test(self, input, epoch=0, save_every=1):
    '''
    Return decoded output.
    '''

    # Note: option 1: time encoding + selection different modules
    '''Encode'''
    input_score, input_coord, input_reshaped = format_input(input, 0, self.shape, save_test=False)

    input = Variable(input_score.cuda())
    input_reshaped = Variable(input_reshaped.cuda())

    batch_size, n_chan, n_dim, traj_length = input_reshaped.size()
    numel = batch_size * traj_length * n_dim
    res_dict = {}

    # Note: option 1: time encoding + selection different modules
    '''Encode'''
    encoder_input = (input , input_reshaped)
    soft_coord = self.temporal_encoding(encoder_input)

    # # Give ground truth to the first point.
    # soft_coord = torch.cat([input_reshaped[..., 0:1, 0:1],soft_coord[..., 1:]], dim=-1)

    '''Losses'''
    #  Note: Option 1: loss_dynamics iteratively
    # loss_dynamics = torch.FloatTensor((-1e7,)).cuda().repeat(soft_coord.shape[0])
    # for t_init in range(self.T-self.subtraj_length+1):
    #   loss_dynamics = torch.max(loss_dynamics, self.loss_dynamics(soft_coord.squeeze(-2)[..., t_init:t_init+self.subtraj_length])).mean()
    # loss_dynamics = loss_dynamics.mean()
    # loss_dynamics = self.loss_dynamics(soft_coord.squeeze(-2))

    #  Note: Option 2: loss_dynamics at once
    loss_dynamics = self.loss_dynamics_jbld(soft_coord.squeeze(-2))
    res_dict['Logdet_G'] = loss_dynamics.item()

    loss_1nn = self.loss_1nn(soft_coord, input_reshaped)
    res_dict['Entropy'] = loss_1nn.item()

    '''Optimizer step'''
    loss = loss_dynamics + self.weight_1nn * loss_1nn
    res_dict['Total_loss'] = loss.item()

    '''Other Losses'''
    # if epoch % save_every:
    #   res_dict['Plot'] = self.save_visuals(gt, M, man, epoch)

    return res_dict

  def update_hyperparameters(self, epoch, n_epochs):

    lr_dict = super(DynClass, self).update_hyperparameters(epoch, n_epochs)

    return lr_dict

  def mask_frames(self, F):
    F_masked = F
    return F_masked

def quick_represent(all, selected, name):
  # for i in range(all.shape[0]):
  #   plt.plot(all[i].detach().cpu().numpy())
  t = np.arange(0,all.shape[1])
  for i in range(all.shape[0]):
    plt.scatter(t, all[i].detach().cpu().numpy())
  plt.plot(selected.detach().cpu().numpy(), 'gx--')
  plt.savefig('results/' + name)
  plt.close()



