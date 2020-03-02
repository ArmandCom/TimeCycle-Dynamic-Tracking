import argparse
import os


class BaseArgs:
  '''
  Arguments for data, model, and checkpoints.
  '''
  def __init__(self):
    self.is_train, self.split = None, None
    self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # hardware
    self.parser.add_argument('--n_workers', type=int, default=8, help='number of threads')
    self.parser.add_argument('--gpus', type=str, default='2', help='visible GPU ids, separated by comma')

    # data
    self.parser.add_argument('--dset_dir', type=str, default=os.path.join('/data/Armand/', 'TimeCycle/'))
    self.parser.add_argument('--generate_dset', type=str, default=False)
    self.parser.add_argument('--dset_name', type=str, default='traj_multi')

    # model
    self.parser.add_argument('--model', type=str, default='crop', help='Model name')

    # dimensions
    self.parser.add_argument('--k', type=int, nargs='+', default=6)
    self.parser.add_argument('--feat_latent_size', type=int, default=256,
                             help='Size of convolutional features')
    self.parser.add_argument('--image_size', type=int, nargs='+', default=[64, 64])

    self.parser.add_argument('--time_enc_size', type=int, default=4,
                             help='Size of temporal encoding')
    self.parser.add_argument('--manifold_size', type=int, default=3,
                             help='Dimension of the manifold for the given time sequence')

    # Changing hyperparameters
    self.parser.add_argument('--traj_length', type=int, default=9)

    # self.parser.add_argument('--weight_dim', type=float, default=0.001,
    #                          help='Weight of the manifold dimension loss - alpha')

    self.parser.add_argument('--ckpt_name', type=str, default='forked_multi_w-noise', help='checkpoint name')

    # ckpt and logging
    self.parser.add_argument('--ckpt_dir', type=str, default=os.path.join('/data/Armand/', 'TimeCycle/'),
                             help='the directory that contains all checkpoints')
    self.parser.add_argument('--log_every', type=int, default=30, help='log every x steps')
    self.parser.add_argument('--save_every', type=int, default=5, help='save every x epochs')
    self.parser.add_argument('--evaluate_every', type=int, default=-1, help='evaluate on val set every x epochs')


  def parse(self):
    opt = self.parser.parse_args()

    # for convenience
    opt.is_train, opt.split = self.is_train, self.split
    opt.dset_path = os.path.join(opt.dset_dir, opt.dset_name)
    if opt.is_train:
      ckpt_name = 'bt{:d}_{:s}'.format(
                      opt.batch_size, opt.ckpt_name)
    else:
        ckpt_name = opt.ckpt_name
    opt.ckpt_path = os.path.join(opt.ckpt_dir, opt.dset_name, ckpt_name)

    # Hard code
    if opt.dset_name == 'traj_multi':
      self.k = 6
      opt.n_channels = 1
      opt.image_size = (1, self.k)
    else:
      raise NotImplementedError

    log = ['Arguments: ']
    for k, v in sorted(vars(opt).items()):
      log.append('{}: {}'.format(k, v))

    return opt, log
