from collections import defaultdict
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .base_model import BaseModel
from models.networks.mapping import Encoder
from models.networks.inv_mapping import Decoder
# from models.networks.encoder import ImageEncoder
# from models.networks.decoder import ImageDecoder
from utils import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO: trick! Switch from run to debug and vice versa --> hold Shift.

class DynClass(BaseModel):

  def __init__(self, opt):
    super(DynClass, self).__init__()

    self.alpha = opt.weight_dim
    self.beta = opt.weight_local_geom
    self.gamma = opt.weight_lin

    self.eps = opt.slack_iso
    self.delta = 1e-2

    self.is_train = opt.is_train
    self.image_size = opt.image_size[0]

    # Data parameters
    # self.__dict__.update(opt.__dict__)
    self.n_channels = opt.n_channels
    self.batch_size = opt.batch_size
    self.n_frames_input = opt.n_frames_input

    # Dimensions
    self.feat_latent_size = opt.feat_latent_size
    self.time_enc_size = opt.time_enc_size
    self.t_enc_rnn_hidden_size = opt.t_enc_rnn_hidden_size
    self.t_enc_rnn_output_size = opt.t_enc_rnn_output_size
    self.manifold_size = opt.manifold_size

    self.ngf = opt.ngf
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

    # Mapping
    encoder_model = Encoder()
    self.encoder_model = nn.DataParallel(encoder_model.cuda())
    self.nets['encoder_model'] = self.encoder_model

    # Inverse Mapping
    decoder_model = Decoder(self.n_frames_input, self.n_frames_output, self.n_channels,
                         self.image_size, self.feat_latent_size,
                         self.ngf, self.manifold_size)
    self.encoder_model = nn.DataParallel(encoder_model.cuda())
    self.decoder_model = nn.DataParallel(decoder_model.cuda())
    self.nets['decoder_model'] = self.decoder_model

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
                     [{'params': self.encoder_model.parameters(), 'lr': self.lr_init}
                     ], betas=(0.9, 0.999))

  def encode(self, input):
    return self.encoder_model(input)

  def decode(self, latent):
    return self.decoder_model(latent)

  def train(self, input, output, neigh, ori_dist, man):

    input = Variable(input.cuda(), requires_grad=False)
    batch_size, n_frames_input, n_dim = input.size()

    numel = batch_size * n_frames_input * n_dim
    loss_dict = {}

    '''Encode'''
    encoder_input = input
    x = self.encode(encoder_input)

    '''Classify'''

    '''Losses'''
    # loss_dict['xxx'] = 0

    '''Optimizer step'''
    loss = 0
    loss_dict['Total_loss'] = loss.item()
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    return loss_dict

  def test(self, input, output, neigh, ori_dist, man, epoch=0, save_every=1):
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

  def save_visuals(self, input, output, output_gt, epoch):
    '''
    Save results. Draw bounding boxes on each component.
    '''
    M_plot = output.detach().cpu().numpy()
    inp_plot = input.detach().cpu().numpy()
    M_gt_plot = output_gt.numpy()

    plt.close()

    elev = 60
    angle = 45

    fig = plt.figure()
    ax1 = fig.add_subplot(231, projection='3d')
    # ax1.set_xlim(-5, 5)
    # ax1.set_ylim(-5, 5)
    #ax1.set_zlim(0, 16)
    ax1.view_init(elev=45, azim=0)
    ax1.plot(M_plot[0,:,0],M_plot[0,:,1], M_plot[0,:,2])
    ax1.scatter(M_plot[0,:,0],M_plot[0,:,1], M_plot[0,:,2],
                c=np.linspace(0, 1, M_plot.shape[1]))

    ax2 = fig.add_subplot(232, projection='3d')
    # ax2.set_xlim(-5, 5)
    # ax2.set_ylim(-5, 5)
    #ax1.set_zlim(0, 16)
    ax2.view_init(elev=-45, azim=0)
    ax2.plot(M_plot[0,:,0],M_plot[0,:,1], M_plot[0,:,2])
    ax2.scatter(M_plot[0,:,0],M_plot[0,:,1], M_plot[0,:,2],
                c=np.linspace(0, 1, M_plot.shape[1]))

    ax3 = fig.add_subplot(233)
    # ax3.set_xlim(-2, 2)
    # ax3.set_ylim(-2, 2)
    ax3.plot(M_plot[0,:,0],M_plot[0,:,1],)
    ax3.scatter(M_plot[0,:,0],M_plot[0,:,1],
                c=np.linspace(0, 1, inp_plot.shape[1]))

    ax4 = fig.add_subplot(234, projection='3d')
    ax4.view_init(elev=elev, azim=angle)
    # ax4.set_xlim(-5, 5)
    # ax4.set_ylim(-5, 5)
    ax4.set_zlim(0, 16)

    ax4.plot(inp_plot[0, :, 0],inp_plot[0, :, 1],inp_plot[0, :, 2])
    ax4.scatter(inp_plot[0, :, 0],inp_plot[0, :, 1],inp_plot[0, :, 2],
                c=np.linspace(0, 1, inp_plot.shape[1]))

    ax5 = fig.add_subplot(235)
    # ax5.set_xlim(-2, 2)
    # ax5.set_ylim(-2, 2)
    ax5.plot(M_gt_plot[0, :, 0], M_gt_plot[0, :, 1])
    ax5.scatter(M_gt_plot[0, :, 0],M_gt_plot[0, :, 1],
                c=np.linspace(0, 1, inp_plot.shape[1]))
    # fig.savefig('test_ep' + str(epoch) + '.png')
    return fig

    # super(LDE, self).save_visuals(gt, output, latent)

  def update_hyperparameters(self, epoch, n_epochs):
    '''
    If when_to_predict_only > 0 and it halfway through training, then only train with
    prediction loss.
    '''
    lr_dict = super(LDE, self).update_hyperparameters(epoch, n_epochs)

    if self.when_to_predict_only > 0 and epoch > int(n_epochs * self.when_to_predict_only):
      self.predict_loss_only = True

    return lr_dict


# def normalize(x, eps=1e-5):
#
#     # seq_norm = nn.BatchNorm1d(x.size(1), affine=False).cuda(gpu_id)
#     mean = torch.mean(x, 1).unsqueeze(1)
#     std = x.std(1).unsqueeze(1) + eps
#
#     x = (x - mean).div(std)
#
#     return x, mean, std
#
# def denormalize(x, mean, std):
#
#     x = (x * std) + mean
#
#     return x