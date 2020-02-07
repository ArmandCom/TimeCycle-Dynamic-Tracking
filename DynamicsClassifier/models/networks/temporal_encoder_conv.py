import torch
import torch.nn as nn
import torch.functional as F

class TemporalEncoder(nn.Module):

    def __init__(self, is_train, input_size, output_size=2048, kernel_width=3, n_layers=3):
        super(TemporalEncoder, self).__init__()

        """
        Each block has 2 convs.
        """
        self.n_layers = n_layers

        layers = []

        self.subnets = [] * n_layers
        for i in range(n_layers):

            if i == n_layers-1:
                hidden_size = output_size
            elif i == 0:
                hidden_size = input_size*2
            else:
                hidden_size *=2

            self.subnets.append(nn.Sequential(
                *self.build_subnet(
                    input_size,
                    hidden_size,
                    kernel_width
                )))

    def forward(self, x):

        # NCT -> NCT1 --> C=1?
        x = x.unsqueeze(-1)

        for i in range(self.n_layers):
            h = self.subnets[i](x)
            # NT1C -> NTC
            h = h.squeeze(-1)
            # skip connection
            x = x+h

        return x

    def build_subnet(self, input_size, output_size, kernel_width):

        if self.use_groupnorm:
            layers =  [nn.GroupNorm(num_groups=32, num_channels=1)] # num groups same as in TF default function
        else:
            layers =  [nn.BatchNorm2d(input_size)]

        layers += [nn.ReLU(inplace=True),
                   nn.Conv2d(in_channels=input_size,
                             out_channels=output_size,
                             kernel_size=[kernel_width,1],
                             padding=int(kernel_width//2))]

        if self.use_groupnorm:
            layers += [nn.GroupNorm(num_groups=32, num_channels=1)]
        else:
            layers += [nn.BatchNorm2d(output_size)]

        layers += [nn.ReLU(inplace=True),
                   nn.Conv2d(in_channels=input_size,
                             out_channels=output_size,
                             kernel_size=[kernel_width,1],
                             padding=int(kernel_width//2))] # initialize with Xavier

        return layers


# def az_fc_block2(net_input, num_filter, kernel_width, is_training,
#                  use_groupnorm=False, name=None):
#     """
#     Full convolutions not separable!!
#     same as az_fc_block, but residual connections is proper Kaiming style
#     of BN -> Relu -> Weight -> BN -> Relu -> Weight -> Add
#     """
#     # NTC -> NT1C
#     net_input_expand = tf.expand_dims(net_input, axis=2)
#     if use_groupnorm:
#         # group-norm
#         net_norm = tf.contrib.layers.group_norm(
#             net_input_expand,
#             channels_axis=-1,
#             reduction_axes=(-3, -2),
#             scope='AZ_FC_block_preact_gn1' + name,
#             reuse=None,
#         )
#     else:
#         # batchnorm
#         net_norm = tf.contrib.layers.batch_norm(
#             net_input_expand,
#             scope='AZ_FC_block_preact_bn1' + name,
#             reuse=None,
#             is_training=is_training,
#         )
#     # relu
#     net_relu = tf.nn.relu(net_norm)
#     # weight
#     net_conv1 = tf.contrib.layers.conv2d(
#         inputs=net_relu,
#         num_outputs=num_filter,
#         kernel_size=[kernel_width, 1],
#         stride=1,
#         padding='SAME',
#         data_format='NHWC',  # was previously 'NCHW',
#         rate=1,
#         activation_fn=None,
#         scope='AZ_FC_block2_conv1' + name,
#         reuse=None,
#     # norm
#     if use_groupnorm:
#         # group-norm
#         net_norm2 = tf.contrib.layers.group_norm(
#             net_conv1,
#             channels_axis=-1,
#             reduction_axes=(-3, -2),
#             scope='AZ_FC_block_preact_gn2' + name,
#             reuse=None,
#         )
#     else:
#         net_norm2 = tf.contrib.layers.batch_norm(
#             net_conv1,
#             scope='AZ_FC_block_preact_bn2' + name,
#             reuse=None,
#             is_training=is_training,
#         )
#
#     # relu
#     net_relu2 = tf.nn.relu(net_norm2)
#
#     # weight
#     small_xavier = variance_scaling_initializer(
#         factor=.001, mode='FAN_AVG', uniform=True)
#
#     net_final = tf.contrib.layers.conv2d(
#         inputs=net_relu2,
#         num_outputs=num_filter,
#         kernel_size=[kernel_width, 1],
#         stride=1,
#         padding='SAME',
#         data_format='NHWC',
#         rate=1,
#         activation_fn=None,
#         weights_initializer=small_xavier,
#         scope='AZ_FC_block2_conv2' + name,
#         reuse=None,
#     )
#
#     # NT1C -> NTC
#     net_final = tf.squeeze(net_final, axis=2)
#     # skip connection
#     residual = tf.add(net_final, net_input)
#
#     return residual
#
#
# def az_fc2_groupnorm(is_training, net, num_conv_layers):
#     """
#     Each block has 2 convs.
#     So:
#     norm --> relu --> conv --> norm --> relu --> conv --> add.
#     Uses full convolution.
#     Args:
#     """
#     for i in range(num_conv_layers):
#         net = az_fc_block2(
#             net_input=net,
#             num_filter=2048,
#             kernel_width=3,
#             is_training=is_training,
#             use_groupnorm=True,
#             name='block_{}'.format(i),
#         )
#     return net