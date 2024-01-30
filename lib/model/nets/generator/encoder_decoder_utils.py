# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch.nn as nn
import torch
import numpy as np
import torchvision
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

##########################################
'''
3D Layers
'''
##########################################
'''
Define a Up-sample block
Network:
  x c*d*h*w->
  -> Up-sample s=2
  -> 3*3*3*c1 or 1*1*1*c1 stride=1 padding conv
  -> norm_layer, activation
'''
class Upsample_3DUnit(nn.Module):
  def __init__(self, kernel_size, input_channel, output_channel, norm_layer, scale_factor=2, upsample_mode='nearest', activation=nn.ReLU(True), use_bias=True):
    super(Upsample_3DUnit, self).__init__()
    if upsample_mode == 'trilinear' or upsample_mode == 'nearest':
      self.block = Upsample_3DBlock(kernel_size, input_channel, output_channel, norm_layer, scale_factor, upsample_mode, activation, use_bias)
    elif upsample_mode == 'transposed':
      self.block = Upsample_TransposedConvBlock(kernel_size, input_channel, output_channel, norm_layer, scale_factor, activation, use_bias)
    else:
      raise NotImplementedError()

  def forward(self, input):
    return self.block(input)


class Upsample_3DBlock(nn.Module):
  def __init__(self, kernel_size, input_channel, output_channel, norm_layer,
               scale_factor=2, upsample_mode='nearest', activation=nn.ReLU(True), use_bias=True):
    super(Upsample_3DBlock, self).__init__()
    conv_block = []
    conv_block += [nn.Upsample(scale_factor=scale_factor, mode=upsample_mode),
                   nn.Conv3d(input_channel, input_channel, kernel_size=kernel_size, groups=input_channel,padding=int(kernel_size//2), bias=use_bias),
                   nn.Conv3d(input_channel, output_channel, kernel_size=1, padding=0, bias=use_bias),
                   norm_layer(output_channel),
                   activation]
    self.block = nn.Sequential(*conv_block)

  def forward(self, input):
    return self.block(input)


class Upsample_TransposedConvBlock(nn.Module):
  def __init__(self, kernel_size, input_channel, output_channel, norm_layer, scale_factor=2, activation=nn.ReLU(True), use_bias=True):
    super(Upsample_TransposedConvBlock, self).__init__()
    conv_block = []
    conv_block += [
      nn.ConvTranspose3d(input_channel, output_channel, kernel_size=kernel_size, padding=int(kernel_size//2), bias=use_bias, stride=scale_factor, output_padding=int(kernel_size//2)),
      norm_layer(output_channel),
      activation
    ]
    self.block = nn.Sequential(*conv_block)

  def forward(self, input):
    return self.block(input)


'''
Define a resnet block
Network:
  x ->
  -> 3*3 stride=1 padding conv
  -> norm_layer
  -> activation
  -> 3*3 stride=1 padding conv
  -> norm_layer

'''
class Resnet_3DBlock(nn.Module):
  def __init__(self, dim, norm_layer,
               activation=nn.ReLU(True), use_bias=True):
    super(Resnet_3DBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim, norm_layer, activation, use_bias)

  def build_conv_block(self, dim, norm_layer,
                       activation, use_bias):
    conv_block = []

    conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=use_bias),
                   norm_layer(dim),
                   activation]

    conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=use_bias),
                   norm_layer(dim)]

    return nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out


'''
Define a p3d ResNet Block
First learn from spatial message and then depth message 
Network:
  x ->
  -> 1*1*1 * tc stride=1 conv 
  -> norm_layer, activation
  -> 1*3*3 * tc stride=1 padding0,1,1 conv
  -> norm_layer, activation
  -> 3*1*1 * tc stride=1 padding1,0,0 conv
  -> norm_layer, activation
  -> 1*1*1 * c stride=1 conv
  -> norm_layer
'''
class Resnet_ST3DBlock(nn.Module):
  def __init__(self, dim, expand_factor, norm_layer, activation, use_bias):
    super(Resnet_ST3DBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim, expand_factor, norm_layer, activation, use_bias)

  def build_conv_block(self, dim, expand_factor, norm_layer, activation=nn.ReLU(True), use_bias=True):
    conv_block = []
    expand_dim = int(expand_factor*dim)
    conv_block += [nn.Conv3d(dim, expand_dim, kernel_size=(1,1,1), padding=0, bias=use_bias),
                   norm_layer(expand_dim),
                   activation]
    conv_block += [nn.Conv3d(expand_dim, expand_dim, kernel_size=(1,3,3), padding=(0,1,1), bias=use_bias),
                   norm_layer(expand_dim),
                   activation,
                   nn.Conv3d(expand_dim, expand_dim, kernel_size=(3,1,1), padding=(1,0,0), bias=use_bias),
                   norm_layer(expand_dim),
                   activation]
    conv_block += [nn.Conv3d(expand_dim, dim, kernel_size=(1, 1, 1), padding=0, bias=use_bias),
                   norm_layer(dim)]
    return nn.Sequential(*conv_block)

  def forward(self, input):
    return input + self.conv_block(input)


'''
Define a 3D Dense block
Network:
  x ->
  -> for i in num_layers:
        -> norm relu 1*1*tg conv 
        -> norm relu 3*3*g conv
      norm
'''
class _DenseLayer3D(nn.Sequential):
  def __init__(self, num_input_features, growth_rate, up_sample, bn_size, norm_layer, activation, use_bias=True):
    super(_DenseLayer3D, self).__init__()
    self.up_sample = up_sample
    self.add_module('norm1', norm_layer(num_input_features)),
    self.add_module('relu1', activation),
    self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                        growth_rate, kernel_size=1, stride=1, bias=use_bias)),
    self.add_module('norm2', norm_layer(bn_size * growth_rate)),
    self.add_module('relu2', activation),
    self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                        kernel_size=3, stride=1, padding=1, bias=use_bias)),

  def forward(self, x):
    new_features = super(_DenseLayer3D, self).forward(x)
    if self.up_sample:
      return new_features
    else:
      return torch.cat([x, new_features], 1)

class Dense_3DBlock(nn.Module):
  def __init__(self, num_layers, num_input_features, up_sample=False, bn_size=4, growth_rate=16, norm_layer=nn.BatchNorm3d, activation=nn.ReLU(True), use_bias=True):
    super(Dense_3DBlock, self).__init__()
    self.up_sample = up_sample
    self.num_input_features = num_input_features
    conv_block = []
    for i in range(num_layers):
      conv_block += [_DenseLayer3D(num_input_features + i * growth_rate, growth_rate, up_sample, bn_size, norm_layer, activation, use_bias)]
    if up_sample:
      self.conv_block = nn.ModuleList(conv_block)
    else:
      self.conv_block = nn.Sequential(*conv_block)
    self.next_input_features = num_input_features + num_layers*growth_rate
    # self.conv_block.add_module('finalnorm', norm_layer(self.next_input_features))

  def forward(self, input):
    if self.up_sample:
      # if up_sample, final output is concatination of all layers'output
      x = input
      out_list = []
      for layer in self.conv_block:
        out = layer(x)
        x = torch.cat([x, out], 1)
        out_list.append(out)
      return torch.cat(out_list, 1)
    else:
      return self.conv_block(input)


##########################################
'''
2D Layers
'''
##########################################
'''
Define a resnet block
Network:
  x ->
  -> 3*3 stride=1 padding conv
  -> norm_layer
  -> activation
  -> 3*3 stride=1 padding conv
  -> norm_layer

'''
class ResnetBlock(nn.Module):
  def __init__(self, dim, norm_layer, activation=nn.ReLU(True), use_dropout=False, use_bias=True):
    super(ResnetBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim, norm_layer, activation, use_bias)

  def build_conv_block(self, dim, norm_layer, activation, use_bias):
    conv_block = []

    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias),
                   norm_layer(dim),
                   activation]

    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias),
                   norm_layer(dim)]

    return nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out

class ResnetBottleneckBlock(nn.Module):
  def __init__(self, dim, norm_layer, activation=nn.ReLU(True), use_dropout=False, use_bias=True):
    super(ResnetBottleneckBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim, norm_layer, activation, use_bias)

  def build_conv_block(self, dim, norm_layer, activation, use_bias):
    conv_block = []

    conv_block += [nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0, bias=use_bias),
                   norm_layer(dim // 4),
                   activation,
                   nn.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, bias=use_bias),
                   norm_layer(dim // 4),
                   activation,
                   nn.Conv2d(dim // 4, dim, kernel_size=1, padding=0, bias=use_bias),
                   norm_layer(dim)
                   ]

    return nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out

'''
Define a 2D Dense block
Network:
  x ->
  -> for i in num_layers:
        -> norm relu 1*1*tg conv 
        -> norm relu 3*3*g conv
      norm
'''
#>>>>>>>>>>>>>>>>>
class _DenseLayer2D(nn.Sequential):
  def __init__(self, num_input_features, growth_rate, bn_size, norm_layer, activation, use_bias):
    super(_DenseLayer2D, self).__init__()
    self.add_module('norm1', norm_layer(num_input_features)),
    self.add_module('relu1', activation),
    self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                        growth_rate, kernel_size=1, stride=1, bias=use_bias)),
    self.add_module('norm2', norm_layer(bn_size * growth_rate)),
    self.add_module('relu2', activation),
    #self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
    #                                    kernel_size=3, stride=1, padding=1, bias=use_bias,groups=growth_rate)),
    self.add_module('conv2_dw', nn.Conv2d(bn_size * growth_rate, bn_size * growth_rate,
                                        kernel_size=3, stride=1, padding=1, groups=bn_size * growth_rate, bias=use_bias)),
    self.add_module('conv2_pw', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=1, stride=1, bias=use_bias)),

  def forward(self, x):
    #print('_DenseLayer2D-x:',x.shape)
    #B, L, C  = x.shape
    
    #x=x.reshape(B, C, -1, int(L**0.5))
    new_features  = super(_DenseLayer2D, self).forward(x)

    #print('new_features:',new_features.shape)
    x=torch.cat([x, new_features], 1)

    #grid_img=torchvision.utils.make_grid(x.squeeze(0).cpu(),nrow=8,normalize=True)
    #plt.imshow(grid_img.permute(1,2,0)[:,:,32].cpu().detach().numpy(),cmap='gray',vmin=0,vmax=1)
    #plt.show()
    #print('x:',x.shape)
    return x

class _DenseBlock2D_Transition(nn.Sequential):
  def __init__(self, num_input_features, num_output_features, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True), use_bias=True):
    super(_DenseBlock2D_Transition, self).__init__()
    self.add_module('norm', norm_layer(num_input_features))
    self.add_module('relu', activation)
    self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                      kernel_size=3, stride=2, padding=1, bias=use_bias))

class Dense_2DBlock(nn.Module):
  def __init__(self, num_layers, num_input_features, bn_size=4, growth_rate=16, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True), use_bias=True):
    super(Dense_2DBlock, self).__init__()
    conv_block = []
    for i in range(num_layers):
      conv_block += [

        _DenseLayer2D(num_input_features + i * growth_rate, growth_rate, bn_size, norm_layer, activation, use_bias)
        
        ]
    self.conv_block = nn.Sequential(*conv_block)
    self.next_input_features = num_input_features + num_layers*growth_rate
    # self.conv_block.add_module('finalnorm', norm_layer(self.next_input_features))

  def forward(self, input):
    x=self.conv_block(input)
    #next_input_features=self.next_input_features
    #print('Dense_2DBlock_x:',x.shape)
    #print('next_input_features:',next_input_features)
    return x

class MR_Dense_2DBlock(nn.Module):
  def __init__(self, num_layers, num_input_features, bn_size=4, growth_rate=16, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True), use_bias=True):
    super(MR_Dense_2DBlock, self).__init__()
    conv_block = []
    encoder_block_list = [6, 12, 24, 16, 6]
    self.conv_0=nn.Conv2d(num_input_features, num_input_features*4, kernel_size=1, stride=1, padding=0, bias=use_bias)
    self.norm_0=nn.BatchNorm2d(num_input_features*4)
    self.conv_block_0=Dense_2DBlock(encoder_block_list[1], num_input_features, bn_size, growth_rate, norm_layer, activation, use_bias)


    self.conv_1=nn.Conv2d(num_input_features*4, num_input_features, kernel_size=1, stride=1, padding=0, bias=use_bias)
    self.norm_1=nn.BatchNorm2d(num_input_features*4)

    self.conv_block_1=Dense_2DBlock(encoder_block_list[2], num_input_features*4, bn_size, growth_rate, norm_layer, activation, use_bias)
    
    


    # self.conv_block.add_module('finalnorm', norm_layer(self.next_input_features))

  def forward(self, x):

    x1_1=self.conv_block_0(x)
    print('x1_1.shape',x1_1.shape)
    x1_2=self.conv_0(x)
    x1_2=self.norm_0(x1_2)
    print('x1_2.shape',x1_2.shape)
    x1=x1_1+x1_2

    x2_1=self.conv_block_1(x1)
    print('x2_1.shape',x2_1.shape)

    return x


class DenseBlock2D_Transition(nn.Module):
  def __init__(self, num_input_features, num_output_features, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True), use_bias=True):
    super(DenseBlock2D_Transition, self).__init__()
    self.conv_block = _DenseBlock2D_Transition(num_input_features, num_output_features, norm_layer, activation, use_bias)

  def forward(self, input):
    return self.conv_block(input)


##########################################
'''
2D To 3D Layers
'''
##########################################
#>>>>>>>>>>>>>>>>>>>>>>
class Dimension_UpsampleCutBlock(nn.Module):
  def __init__(self, input_channel, output_channel, norm_layer2d, norm_layer3d, activation=nn.ReLU(True), use_bias=True):
    super(Dimension_UpsampleCutBlock, self).__init__()

    self.output_channel = output_channel
    compress_block = [
      nn.Conv2d(input_channel, output_channel, kernel_size=1, padding=0, bias=use_bias),
      norm_layer2d(output_channel),
      activation
    ]
    self.compress_block = nn.Sequential(*compress_block)

    conv_block = []
    conv_block += [
      #nn.Conv3d(output_channel, output_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=use_bias),
      nn.Conv3d(output_channel, output_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=output_channel, bias=use_bias),
      nn.Conv3d(output_channel, output_channel, kernel_size=1, padding=0, bias=use_bias),
      norm_layer3d(output_channel),
      activation,
    ]
    self.conv_block = nn.Sequential(*conv_block)

  def forward(self, input):
    # input's shape is [NCHW]
    N,_,H,W = input.size()
    # expand to [NCDHW]
    return self.conv_block(self.compress_block(input).unsqueeze(2).expand(N,self.output_channel,H,H,W))


class Dimension_UpsampleBlock(nn.Module):
  def __init__(self, input_channel, output_channel, norm_layer, activation=nn.ReLU(True), use_bias=True):
    super(Dimension_UpsampleBlock, self).__init__()
    conv_block = []
    conv_block += [
      nn.Conv3d(input_channel, input_channel, kernel_size=(3,1,1), padding=(1,0,0), bias=use_bias),
      norm_layer(input_channel),
      activation,
      nn.Conv3d(input_channel, output_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=use_bias),
      norm_layer(output_channel),
      activation,
    ]
    self.conv_block = nn.Sequential(*conv_block)

  def forward(self, input):
    # input's shape is [NCHW]
    N,C,H,W = input.size()
    # expand to [NCDHW]
    return self.conv_block(input.unsqueeze(2).expand(N,C,H,H,W))

class Dimension_UpsampleBlockSuper(nn.Module):
  def __init__(self, input_channel, output_channel, norm_layer, activation=nn.ReLU(True), use_bias=True):
    super(Dimension_UpsampleBlockSuper, self).__init__()
    conv_block = []
    inner_channel = int(input_channel//2)
    conv_block += [
      nn.Conv3d(input_channel, input_channel, kernel_size=(3,1,1), padding=(1,0,0), bias=use_bias),
      norm_layer(input_channel),
      activation,
      nn.Conv3d(input_channel, inner_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=use_bias),
      norm_layer(inner_channel),
      activation,
      nn.Conv3d(inner_channel, inner_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=use_bias),
      norm_layer(inner_channel),
      activation,
      nn.Conv3d(inner_channel, output_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=use_bias),
      norm_layer(output_channel),
      activation,
    ]
    self.conv_block = nn.Sequential(*conv_block)

  def forward(self, input):
    # input's shape is [NCHW]
    N,C,H,W = input.size()
    # expand to [NCDHW]-唯独扩展将四维变成五维
    return self.conv_block(input.unsqueeze(2).expand(N,C,H,H,W))

class Dimension_UpsampleBlockSuper1(nn.Module):
  def __init__(self, input_channel, output_channel, norm_layer, activation=nn.ReLU(True), use_bias=True):
    super(Dimension_UpsampleBlockSuper1, self).__init__()
    conv_block = []
    inner_channel = int(input_channel//2)
    conv_block += [
      nn.Conv3d(input_channel, input_channel, kernel_size=(3,1,1), padding=(1,0,0), bias=use_bias),
      norm_layer(input_channel),
      activation,
      nn.Conv3d(input_channel, output_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=use_bias),
      norm_layer(output_channel),
      activation,
      nn.Conv3d(output_channel, output_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=use_bias),
      norm_layer(output_channel),
      activation,
    ]
    self.conv_block = nn.Sequential(*conv_block)

  def forward(self, input):
    # input's shape is [NCHW]
    N,C,H,W = input.size()
    # expand to [NCDHW]
    return self.conv_block(input.unsqueeze(2).expand(N,C,H,H,W))


##########################################
'''
View Fusion Layers / Functions
'''
##########################################
#view1Order=opt.CTOrder_Xray1 CTOrder_Xray1: [0, 1, 3, 2, 4]
#view2Order=opt.CTOrder_Xray2 CTOrder_Xray2: [0, 1, 4, 2, 3]
class Transposed_And_Add(nn.Module):
  def __init__(self, view1Order, view2Order, sortOrder=None):
    super(Transposed_And_Add, self).__init__()
    self.view1Order = view1Order
    self.view2Order = view2Order
    #tuple(...) 将这个 NumPy 数组转换为一个元组。
    #使用 NumPy 库的 argsort 函数对 view1Order 中的元素进行排序，并返回排序后元素的索引。
    #这行代码的目的是将 view1Order 中的元素按升序排序后，返回一个元组。
    #前面使用过view1Order，但是现在的不是我需要的了
    self.permuteView1 = tuple(np.argsort(view1Order))
    self.permuteView2 = tuple(np.argsort(view2Order))
    self.softmax_layer=nn.Softmax(dim=-1)
    '''
    conv_block = []
    activation=nn.ReLU(True)
    norm_layer = nn.InstanceNorm3d
    use_bias=True
    conv_block += [
      nn.ConvTranspose3d(input_channel, output_channel, kernel_size=kernel_size, padding=int(kernel_size//2), bias=use_bias, stride=scale_factor, output_padding=int(kernel_size//2)),
      norm_layer(output_channel),
      activation
    ]
    
    conv_block += [ 
      nn.Conv3d(input_channel, input_channel, kernel_size=(3,1,1), padding=(1,0,0), bias=use_bias),
      norm_layer(input_channel),
      activation,
    ]
    
    self.conv_block = nn.Sequential(*conv_block)
    self.attn_drop = nn.Dropout(0.2)
    '''

  def forward(self, *input):
    x1=input[0].permute(*self.permuteView1)
    x2=input[1].permute(*self.permuteView2)
    '''
    x=x1*x2
    B1,C1,H1,W1,Z1=x.shape
    #print('x1.shape:',x1.shape)
    conv1=nn.ConvTranspose3d(C1, C1, kernel_size=(3,1,1), padding=(1,0,0), bias=False)
    w1=conv1(x.to(conv1.weight.device)).cuda()
    #print('w1.shape:',w1.shape)

    B2,C2,H2,W2,Z2=x.shape
    conv2=nn.ConvTranspose3d(C2, C2, kernel_size=(3,3,3), padding=(1,1,1), bias=False)
    w2=conv1(x.to(conv2.weight.device)).cuda()
    x_f=w1 * x1 +w2 * x2
    '''

    #print('x1:',x1)
    #print('x2.shape:',x2.shape)
    #print('x2:',x2)
    #x=x1 @ x2.transpose(-2,-1)
    #print('x:',x)
    #x=self.softmax_layer(x)
    #x=self.attn_drop(x)
    #x=(x1+x2)*(x1*x2)
    #x_f=x * x1 +x * x2

    # return tensor in order of sortOrder
    return x1+x2