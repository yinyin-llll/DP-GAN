# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import numpy as np
import functools

from lib.model.nets.generator.encoder_decoder_utils import *


'''
Wrapping Architecture
'''

#这是一个名为Link_Encoder_Decoder的自定义神经网络模块（nn.Module的子类）。
#它由一个编码器（encoder）、解码器（decoder）和连接器（linker）组成。
class Link_Encoder_Decoder(nn.Module):

  #初始化方法中，encoder、decoder和linker被传入并保存为该模块的属性。
  def __init__(self, encoder, decoder, linker):
    super(Link_Encoder_Decoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.linker = linker

  #在前向传播方法中，输入input首先经过编码器，然后通过连接器，最后经过解码器。
  #最终的输出结果作为该模块的前向传播结果返回。
  def forward(self, input):
    return self.decoder(self.linker(self.encoder(input)))
  
#这个模块的作用是将输入数据通过编码器进行特征提取，然后通过连接器将编码器的输出与解码器相连接，并通过解码器进行数据重建或生成。



'''
Link Encoder to Decoder
'''

#这是一个名为Linker_FC的自定义神经网络模块（nn.Module的子类）。它是连接器（Linker）的一种实现。
class Linker_FC(nn.Module):

  #在初始化方法中，它接收几个参数：
  # noise_len表示噪声向量的长度
  # encoder_out_channel表示编码器的输出通道数
  # encoder_out_shape表示编码器的输出形状
  # decoder_input_shape表示解码器的输入形状
  # decoder_input_nc表示解码器的输入通道数

  def __init__(self, noise_len, encoder_out_channel, encoder_out_shape, decoder_input_shape, decoder_input_nc):
    super(Linker_FC, self).__init__()

    #在初始化方法中，使用nn.ReLU(True)创建了一个激活函数对象activation。
    activation = nn.ReLU(True)

    # Encoder
    #首先计算了编码器输入的维度encoder_input_cell，该值等于编码器输出的通道数乘以编码器输出的形状的高和宽的乘积。
    encoder_input_cell = encoder_out_shape[0] * encoder_out_shape[1] * encoder_out_channel

    #然后使用nn.Linear创建了一个线性层将encoder_input_cell维度的输入映射到长度为noise_len的向量，并接在其后使用了激活函数activation。
    encoder_fc = [
      nn.Linear(int(encoder_input_cell), noise_len),
      activation]
    # Decoder

    #首先保存了解码器的输入通道数和输入形状为该模块的属性
    self.decoder_input_nc = decoder_input_nc
    self.decoder_input_shape = decoder_input_shape

    #然后计算了解码器输出的维度decoder_output_cell，该值等于解码器输入的通道数乘以解码器输入的形状的高和宽的乘积。
    decoder_output_cell = functools.reduce(lambda x,y:x*y, decoder_input_shape) * decoder_input_nc

    #然后使用nn.Linear创建了一个线性层将长度为noise_len的向量映射到维度为decoder_output_cell的输出，并接在其后使用了激活函数activation。
    decoder_fc = [
      nn.Linear(noise_len, decoder_output_cell),
      activation
    ]

    #最后，通过nn.Sequential将编码器和解码器的层按顺序组合成一个顺序模型，并将其保存为该模块的属性linker。
    print('Link {} to {} to {}'.format(encoder_input_cell, noise_len, decoder_output_cell))
    self.linker = nn.Sequential(*(encoder_fc+decoder_fc))

  def forward(self, input):
    #input.view(input.size(0), -1)将输入张量从任意形状变为形状为 (input.size(0), -1) 的张量
    #其中 input.size(0) 是输入张量的第一个维度的大小，而 -1 则表示将剩余的元素展平
    #这一次，将结果从形状为 (input.size(0), -1) 变为形状为 (input.size(0), self.decoder_input_nc, *self.decoder_input_shape) 的张量，
    #其中 self.decoder_input_nc 和 self.decoder_input_shape 分别是解码器输入的通道数和形状。
    return self.linker(input.view(input.size(0), -1)).view(input.size(0), self.decoder_input_nc, *self.decoder_input_shape)


#这是一个名为Linker_DimensionUp_Conv的自定义神经网络模块（nn.Module的子类）。
#它是连接器（Linker）的一种实现。
class Linker_DimensionUp_Conv(nn.Module):

  #在初始化方法中，它接收几个参数：
  # encoder_out_channel表示编码器的输出通道数
  # encoder_out_shape表示编码器的输出形状
  # decoder_input_shape表示解码器的输入形状
  # decoder_input_nc表示解码器的输入通道数
  # norm_layer表示归一化层的类型
  def __init__(self, encoder_out_channel, encoder_out_shape, decoder_input_shape, decoder_input_nc, norm_layer):
    super(Linker_DimensionUp_Conv, self).__init__()

    #通过判断norm_layer的类型来确定是否使用偏置项。
    #如果norm_layer是functools.partial类型，则判断其内部函数是否为nn.InstanceNorm2d；
    if type(norm_layer) == functools.partial:

      #如果norm_layer是nn.InstanceNorm2d类型，则设置use_bias为True
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:

      #否则，设置use_bias为False
      use_bias = norm_layer == nn.InstanceNorm2d

    #创建一个激活函数对象activation，使用nn.ReLU(True)来表示ReLU激活函数
    activation = nn.ReLU(True)


    #使用Dimension_UpsampleBlock创建一个维度上采样块（Dimension Upsample Block）。
    #该块将编码器的输出特征图的通道数转换为解码器的输入特征图的通道数，同时进行归一化和激活操作。
    #这个维度上采样块被保存为该模块的属性linker。
    self.linker = Dimension_UpsampleBlock(encoder_out_channel, decoder_input_nc, norm_layer, activation, use_bias)

    #在打印语句中显示了连接器的形状变换信息，展示了编码器输出特征图的形状和通道数变换到解码器输入特征图的形状和通道数。
    print('Link shape={}{} to shape={}{}'.format(encoder_out_shape, encoder_out_channel, decoder_input_shape, decoder_input_nc))

  #在前向传播方法中，输入通过连接器linker进行转换，并返回转换后的输出。
  def forward(self, input):
    return self.linker(input)
  
#该模块的作用是通过维度上采样块将编码器的输出特征图的通道数转换为解码器的输入特征图的通道数，用于编码器和解码器之间的连接。
#该连接器可用于实现特征图的尺寸转换和通道数对齐，用于数据的重建或生成任务。


#这是一个名为Linker_FC_multiview的自定义神经网络模块（nn.Module的子类），用于多视角（Multiview）情况下的连接器（Linker）实现
class Linker_FC_multiview(nn.Module):

  #在初始化方法中，它接收多个参数：
  # input_views_n表示输入的视角数
  # noise_len表示噪声的长度
  # encoder_out_channel表示编码器的输出通道数
  # encoder_out_shape表示编码器的输出形状
  # decoder_input_shape表示解码器的输入形状
  # decoder_input_nc表示解码器的输入通道数。
  def __init__(self, input_views_n, noise_len, encoder_out_channel, encoder_out_shape, decoder_input_shape, decoder_input_nc):
    super(Linker_FC_multiview, self).__init__()

    #将input_views_n保存为该模块的属性self.input_views_n
    self.input_views_n = input_views_n

    #创建一个激活函数对象activation，使用nn.ReLU(True)来表示ReLU激活函数
    activation = nn.ReLU(True)
    # Encoder
    encoder_input_cell = encoder_out_shape[0] * encoder_out_shape[1] * encoder_out_channel

    #通过循环创建多个编码器视角。
    #对于每个视角，使用线性层nn.Linear将编码器的输入特征图展平为一维向量，并进行激活操作。
    for i in range(input_views_n):
      encoder_fc = [
        nn.Linear(int(encoder_input_cell), noise_len),
        activation]
      
      #这些编码器视角被保存为该模块的属性，属性名称为'view' + str(i)，其中i为视角的索引
      setattr(self, 'view' + str(i), nn.Sequential(*(encoder_fc)))
    # Fuse module

    #创建融合模块（Fuse module）。
    #该模块将多个编码器视角的输出进行拼接，并通过线性层将拼接后的特征向量转换为长度为noise_len的噪声向量，并进行激活操作。
    fuse_module = [
      nn.Linear(input_views_n * noise_len, noise_len),
      activation]
    
    #这个融合模块被保存为该模块的属性self.fuse_module
    self.fuse_module = nn.Sequential(*fuse_module)

    # Decoder
    self.decoder_input_nc = decoder_input_nc
    self.decoder_input_shape = decoder_input_shape
    decoder_output_cell = functools.reduce(lambda x,y:x*y, decoder_input_shape) * decoder_input_nc

    #解码器的输入通道数为noise_len，通过线性层将噪声向量转换为解码器的输出特征图的展平表示，并进行激活操作。
    #这个解码器部分被保存为该模块的属性self.decoder_fc。
    decoder_fc = [
      nn.Linear(noise_len, decoder_output_cell),
      activation
    ]

    #在打印语句中显示了连接器的形状变换信息，展示了编码器输出特征图的形状和通道数变换到融合模块输出的形状和解码器的输入形状。
    self.decoder_fc = nn.Sequential(*(decoder_fc))
    print('Link {} to {} to {} to {}'.format(encoder_input_cell, 2*noise_len, noise_len, decoder_output_cell))


  #输入参数input为多个视角的输入数据组成的列表，其中每个视角的输入数据为一个张量
  def forward(self, input):

    #首先，根据input_views_n与输入数据列表长度的一致性进行断言判断。
    assert len(input) == self.input_views_n
    out_list = []

    #然后，遍历输入数据列表，将每个视角的输入数据通过相应的编码器视角进行特征提取。
    for index, input_view in enumerate(input):
      out = getattr(self, 'view' + str(index))(input_view.view(input_view.size(0), -1))

      #每个编码器视角的输出被保存到一个列表out_list中
      out_list.append(out)

    #将out_list中的编码器视角输出进行拼接，并通过融合模块self.fuse_module将拼接后的特征向量转换为长度为noise_len的噪声向量
    #将噪声向量通过解码器部分self.decoder_fc进行解码，得到解码器的输出特征图
    #最终，将解码器的输出特征图进行形状调整，以匹配解码器的输入形状，并返回结果
    return self.decoder_fc(self.fuse_module(torch.cat(out_list, dim=1))).view(input_view.size(0), self.decoder_input_nc, *self.decoder_input_shape)


'''
Encoder
'''

#这是一个名为Conv_connect_2D_encoder的自定义神经网络模块（nn.Module的子类），用于2D卷积连接器的编码器部分的实现。
class Conv_connect_2D_encoder(nn.Module):

  #在初始化方法中，它接收多个参数：
  # input_shape表示输入的图像形状（高度和宽度）
  # input_nc表示输入图像的通道数
  # ngf表示编码器的基础通道数
  # norm_layer表示归一化层的类型（默认为nn.BatchNorm2d）
  # n_downsampling表示下采样的次数
  # n_blocks表示每个下采样阶段的残差块数量。
  def __init__(self, input_shape=(128, 128), input_nc=1, ngf=16, norm_layer=nn.BatchNorm2d,
               n_downsampling=4, n_blocks=3):
    super(Conv_connect_2D_encoder, self).__init__()

    #首先，通过断言判断n_blocks必须大于等于0
    assert (n_blocks >= 0)

    # 根据norm_layer的类型确定是否使用偏置项use_bias，
    # 其中根据norm_layer是否为functools.partial类型进行判断。
    if type(norm_layer) == functools.partial:
      #true
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
      #false
      use_bias = norm_layer == nn.InstanceNorm2d

    #创建激活函数对象activation，使用nn.ReLU(True)来表示ReLU激活函数。
    activation = nn.ReLU(True)

    #创建一个空列表model，用于存储编码器的层。
    model = []

    out_c = input_nc
    for i in range(2):
      in_c = out_c
      out_c = 3

      #对于input_nc通道的输入图像，先进行反射填充，然后使用3x3的卷积核进行卷积操作，输出通道数为3，接着进行归一化和激活操作。
      model += [nn.ReflectionPad2d(1),
               nn.Conv2d(in_c, out_c, kernel_size=3, padding=0, bias=use_bias),
               norm_layer(out_c),
               activation]
  
    #使用3x3的卷积核进行卷积操作，输出通道数为ngf，步长为2，进行归一化和激活操作
    model += [nn.ReflectionPad2d(1),
              nn.Conv2d(out_c, ngf, kernel_size=3, padding=0, stride=2, bias=use_bias),
              norm_layer(ngf),
              activation]

    ## downsample

    #根据下采样的次数n_downsampling，进行多次下采样操作。
    for i in range(n_downsampling-1):
      mult = 2 ** i
      for _ in range(n_blocks):

        #使用ResnetBlock构建一个残差块，
        # 输入通道数为ngf * mult，
        # 激活函数为activation，
        # 归一化层为norm_layer，
        # 使用偏置项use_bias，
        # 然后进行激活操作
        # 这一步骤重复n_blocks次。
        model += [ResnetBlock(ngf * mult, activation=activation, norm_layer=norm_layer, use_bias=use_bias),
                  activation]
      
      #通过1x1的卷积核进行卷积操作，将通道数从ngf * mult变为ngf * mult * 2，步长为2，进行归一化和激活操作。
      model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                activation]
      
    #保存了下一层编码器的输入通道数self.next_input_channels
    self.next_input_channels = ngf * 2 ** (n_downsampling - 1)

    #通过encoder_stride计算了下一层特征图的大小self.next_feature_size
    encoder_stride = 2 ** n_downsampling
    self.next_feature_size = (input_shape[0] // encoder_stride, input_shape[1] // encoder_stride)

    #将整个编码器的层保存为顺序容器self.model
    self.model = nn.Sequential(*model)

  #模块的作用是构建2D卷积连接器的编码器部分，将输入图像经过多层卷积和下采样操作，得到编码后的特征图。

  #@property装饰器用于定义类的属性，使其可以像访问实例属性一样直接访问，而不需要调用方法。

  @property

  #OutChannels属性是一个只读属性，用于获取编码器输出的通道数。
  #它返回self.next_input_channels，即编码器的下一层输入通道数。
  def OutChannels(self):
    return self.next_input_channels

  
  @property
  #OutFeatureSize属性是一个只读属性，用于获取编码器输出的特征图大小。
  #它返回self.next_feature_size，即编码器的下一层特征图大小。
  def OutFeatureSize(self):
    return self.next_feature_size

  #forward方法定义了前向传播过程。
  #它接收一个输入input，并将其作为参数传递给编码器的顺序容器self.model，然后返回模型的输出结果
  def forward(self, input):
    return self.model(input)

#定义了一个名为DenseConv_connect_2D_encoder的类，它是一个继承自nn.Module的模型。
class DenseConv_connect_2D_encoder(nn.Module):

  #构造函数__init__接受一些参数，
  # 包括输入形状input_shape、
  # 输入通道数input_nc、
  # 中间特征图通道数ngf、
  # 规范化层norm_layer、
  # 下采样次数n_downsampling
  # 残差块的数量n_blocks。
  def __init__(self, input_shape=(128, 128), input_nc=1, ngf=16, norm_layer=nn.BatchNorm2d, n_downsampling=4, n_blocks=3):
    super(DenseConv_connect_2D_encoder, self).__init__()
    assert (n_blocks >= 0)

    #根据参数设置了激活函数activation和是否使用偏置项use_bias
    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = norm_layer == nn.InstanceNorm2d

    activation = nn.ReLU(True)

    model = []

    #定义初始的输入通道数out_c为输入通道数input_nc
    out_c = input_nc

    #通过两个卷积层将输入通道数从input_nc转换为3，采用反射填充和卷积操作，并应用规范化和激活函数
    for i in range(2):
      in_c = out_c
      out_c = 3
      model += [nn.ReflectionPad2d(1),
               nn.Conv2d(in_c, out_c, kernel_size=3, padding=0, bias=use_bias),
               norm_layer(out_c),
               activation]
    
    #使用一个卷积层将通道数从3转换为ngf，采用反射填充、卷积、规范化和激活函数，并实现步长为2的下采样
    model += [nn.ReflectionPad2d(1),
              nn.Conv2d(out_c, ngf, kernel_size=3, padding=0, stride=2, bias=use_bias),
              norm_layer(ngf),
              activation]

    num_layers = 5
    growth_rate = 16
    bn_size = 4
    num_input_channels = ngf
    ## downsample

    #使用多个稠密块和过渡层进行下采样。
    for i in range(n_downsampling):

      #其中，稠密块由多个密集连接的卷积层组成
      model += [
        Dense_2DBlock(num_layers, num_input_channels, bn_size, growth_rate, norm_layer, activation, use_bias)
      ]
      num_input_channels = num_input_channels + num_layers*growth_rate
      if i != n_downsampling-1:
        num_out_channels = num_input_channels // 2

        #过渡层用于减少通道数和特征图的大小
        model += [
          DenseBlock2D_Transition(num_input_channels, num_out_channels, norm_layer, activation, use_bias)
        ]
        num_input_channels = num_out_channels
    model += [

      #最后应用规范化层，得到最终的输出特征图
      norm_layer(num_input_channels)
    ]

    #还设置了next_input_channels作为下一层的输入通道数
    self.next_input_channels = num_input_channels

    #encoder_stride作为编码器的步长
    encoder_stride = 2 ** n_downsampling

    #next_feature_size作为输出特征图的大小
    self.next_feature_size = (input_shape[0] // encoder_stride, input_shape[1] // encoder_stride)
    self.model = nn.Sequential(*model)

  #定义了OutChannels和OutFeatureSize两个属性
  @property
  def OutChannels(self):

    #OutChannels属性返回next_input_channels，即下一层的输入通道数
    return self.next_input_channels

  @property
  def OutFeatureSize(self):

    #OutFeatureSize属性返回next_feature_size，即输出特征图的大小
    return self.next_feature_size

  #定义了forward方法作为前向传播函数。
  #它接收一个输入input，并将其作为参数传递给顺序容器self.model，然后返回模型的输出结果
  def forward(self, input):
    return self.model(input)

#这个类用于构建一个具有稠密连接的卷积编码器，通过多个稠密块和过渡层进行下采样，最终输出特征图的通道数和大小由属性OutChannels和OutFeatureSize提供。


#定义了一个名为Conv_connect_2D_encoder_multiview的类，它是一个继承自nn.Module的模型。
class Conv_connect_2D_encoder_multiview(nn.Module):

  #构造函数__init__接受一些参数
  # 包括输入视图数量input_views
  # 输入形状input_shape
  # 输入通道数input_nc
  # 中间特征图通道数ngf
  # 规范化层norm_layer
  # 下采样次数n_downsampling
  # 残差块的数量n_blocks
  def __init__(self, input_views, input_shape=(128, 128), input_nc=1, ngf=16, norm_layer=nn.BatchNorm2d, n_downsampling=4, n_blocks=3):
    
    super(Conv_connect_2D_encoder_multiview, self).__init__()
    self.input_views = input_views

    for i in range(input_views):
      
      #通过循环创建多个Conv_connect_2D_encoder的实例，并将其设置为类的属性。
      layer = Conv_connect_2D_encoder(input_shape, input_nc, ngf, norm_layer, n_downsampling, n_blocks)
      
      #每个视图对应一个Conv_connect_2D_encoder实例，并通过setattr方法将其命名为view0、view1等。
      setattr(self, 'view'+str(i), layer)

      #同时，通过这些实例获取每个视图的输出通道数和输出特征图大小，并将其分别设置为类的属性next_input_channels和next_feature_size。
      self.next_input_channels = layer.OutChannels
      self.next_feature_size = layer.OutFeatureSize

  #定义了OutChannels和OutFeatureSize两个属性，用于返回多视图编码器的输出通道数和输出特征图大小。
  @property
  def OutChannels(self):
    return self.next_input_channels

  @property
  def OutFeatureSize(self):
    return self.next_feature_size

  #定义了forward方法作为前向传播函数。
  # 它接收一个输入input，其中input是一个列表，包含多个视图的输入数据。
  def forward(self, input):
    assert len(input) == self.input_views
    out_list = []

    #在前向传播过程中，通过循环遍历每个视图的输入数据，并通过对应的Conv_connect_2D_encoder实例进行前向传播
    for index, view in enumerate(input):
      out = getattr(self, 'view'+str(index))(view)

      #将每个视图的输出结果存储在列表out_list中
      out_list.append(out)
    
    #最终，返回out_list作为多视图编码器的输出结果。
    return out_list
# 这个类用于构建一个多视图的卷积编码器，通过多个单视图的Conv_connect_2D_encoder组成
# 并对每个视图的输入数据进行独立的编码过程，最终输出每个视图的编码结果。
# 输出结果由列表out_list存储，列表中的每个元素对应一个视图的编码结果。


#定义了一个名为DenseConv_connect_2D_encoder_multiview的类，它是一个继承自nn.Module的模型。
class DenseConv_connect_2D_encoder_multiview(nn.Module):

  #构造函数__init__接受一些参数
  # 包括输入视图数量input_views
  # 输入形状input_shape
  # 输入通道数input_nc
  # 中间特征图通道数ngf
  # 规范化层norm_layer
  # 下采样次数n_downsampling
  # 残差块的数量n_blocks。
  def __init__(self, input_views, input_shape=(128, 128), input_nc=1, ngf=16, norm_layer=nn.BatchNorm2d, n_downsampling=4, n_blocks=3):

    super(DenseConv_connect_2D_encoder_multiview, self).__init__()
    self.input_views = input_views

    for i in range(input_views):
      #通过循环创建多个DenseConv_connect_2D_encoder的实例，并将其设置为类的属性。
      layer = DenseConv_connect_2D_encoder(input_shape, input_nc, ngf, norm_layer, n_downsampling, n_blocks)

      # 每个视图对应一个DenseConv_connect_2D_encoder实例，并通过setattr方法将其命名为view0、view1等。
      setattr(self, 'view'+str(i), layer)

      #同时，通过这些实例获取每个视图的输出通道数和输出特征图大小
      # 并将其分别设置为类的属性next_input_channels和next_feature_size。
      self.next_input_channels = layer.OutChannels
      self.next_feature_size = layer.OutFeatureSize

  #定义了OutChannels和OutFeatureSize两个属性，用于返回多视图编码器的输出通道数和输出特征图大小。
  @property
  def OutChannels(self):
    return self.next_input_channels

  @property
  def OutFeatureSize(self):
    return self.next_feature_size

  #定义了forward方法作为前向传播函数。它接收一个输入input，其中input是一个列表，包含多个视图的输入数据。
  def forward(self, input):
    assert len(input) == self.input_views
    out_list = []

    #在前向传播过程中，通过循环遍历每个视图的输入数据，并通过对应的DenseConv_connect_2D_encoder实例进行前向传播
    for index, view in enumerate(input):

      #将每个视图的输出结果存储在列表out_list中。
      # 最终，返回out_list作为多视图编码器的输出结果
      out = getattr(self, 'view'+str(index))(view)
      out_list.append(out)
    return out_list

#这个类用于构建一个多视图的密集卷积编码器，通过多个单视图的DenseConv_connect_2D_encoder组成
# 并对每个视图的输入数据进行独立的编码过程，最终输出每个视图的编码结果。
# 输出结果由列表out_list存储，列表中的每个元素对应一个视图的编码结果。

########################################
'''
Decoder
'''
########################################
'''
3DGenerator_Decoder
=> Decoder + out_activation
'''

#定义了一个名为Generator_3DResNetDecoder的类，它是一个继承自nn.Module的模型。
class Generator_3DResNetDecoder(nn.Module):

  #构造函数__init__接受一些参数
  # 包括输入形状input_shape
  # 输入通道数input_nc
  # 输出形状output_shape
  # 输出通道数output_nc
  # 规范化层norm_layer
  # 上采样模式upsample_mode
  # 残差块的数量n_blocks
  # 输出激活函数out_activation。
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DResNetDecoder, self).__init__()
    assert(n_blocks >= 0)
    assert (input_shape[0] == input_shape[1]) and (input_shape[1] == input_shape[2])
    assert (output_shape[0] == output_shape[1]) and (output_shape[1] == output_shape[2])

    self.input_shape = input_shape
    self.input_nc = input_nc

    #在构造函数中，首先进行一些断言检查，确保输入输出形状和通道数的合法性。
    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = norm_layer == nn.InstanceNorm3d

    activation = nn.ReLU(True)

    #接下来，定义了一个空的列表model，用于存储模型的层。
    model = [

      #通过调用build_upsampling_block方法构建了一个上采样块，并将其添加到model列表中。
      # 该上采样块使用输入通道数作为输入和输出通道数，采用规范化层、激活函数和指定的上采样模式，
      # 通过参数up_sample和up_conv来控制是否进行上采样和上卷积操作。
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]


    # #upsample
    max_up = int(np.max(np.log2(np.asarray(output_shape) / np.asarray(input_shape))))
    max_down = int(np.log2(input_nc/output_nc))
    assert max_up < max_down, 'upsampling overstep border'

    out_channel = input_nc
    #根据输入形状和输出形状的比例关系，循环构建多个上采样块，并将它们添加到model列表中。
    #每次循环，输入通道数减半，直到达到输出通道数。
    for i in range(max_up-1):
      in_channel = out_channel
      out_channel = int(in_channel / 2)
      model += [
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    #添加了一个3D卷积层，将最后的输出通道数转换为输出通道数output_nc。
    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
    ]

    # out activation
    #根据参数out_activation指定的激活函数类型，添加了一个输出激活层。
    model += [
      out_activation()
    ]

    #使用nn.Sequential将model列表中的层组合成一个顺序的模型。
    self.model = nn.Sequential(*model)

  #build_upsampling_block方法用于构建上采样块。
  # 它接受一些参数，
  # 包括输入通道数input_nc、
  # 输出通道数output_nc、
  # 规范化层norm_layer、
  # 激活函数activation、
  # 是否进行上采样up_sample、
  # 是否进行上卷积up_conv、
  # 上采样模式upsample_mode、
  # 残差块的数量block_n
  # 是否使用偏置项use_bias
  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    #首先创建一个空的列表blocks，用于存储块的层。
    blocks = []

    #如果up_sample为True，则添加一个上采样层nn.Upsample，将特征图的尺寸上采样两倍。
    # 然后，添加一个3D卷积层nn.Conv3d，将输入通道数input_nc转换为输出通道数output_nc。
    # 接下来，添加规范化层norm_layer和激活函数activation。
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    #如果up_conv为True，则通过循环添加block_n个残差块Resnet_3DBlock和激活函数activation。
    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_3DBlock(output_nc, activation=activation, norm_layer=norm_layer, use_bias=use_bias),
          activation
        ]

    #最后，使用nn.Sequential将blocks列表中的层组合成一个顺序的块，并将其作为结果返回。
    return nn.Sequential(*blocks)

  #forward方法用于前向传播。
  # 它接受输入input作为参数，并将其传递给模型self.model进行计算。
  def forward(self, input):
    # #upsample and convolution
    return self.model(input)

#这个类用于构建一个基于3D ResNet结构的解码器模型。
# 它通过多个上采样块将输入特征图从较小的空间尺寸逐渐上采样到较大的空间尺寸，并最终生成输出特征图。
# 上采样块通过残差块的堆叠来实现特征的逐层恢复和细化。


#定义了Generator_TransposedConvDecoder类，用于实现转置卷积解码器模型。
class Generator_TransposedConvDecoder(nn.Module):

  #构造函数__init__接受一系列参数，
  # 包括输入形状input_shape
  # 输入通道数input_nc
  # 输出形状output_shape
  # 输出通道数output_nc
  # 规范化层norm_layer
  # 上采样模式upsample_mode
  # 残差块数量n_blocks
  # 输出激活函数out_activation。
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_TransposedConvDecoder, self).__init__()

    #在构造函数中，首先进行了一些断言检查，确保输入和输出形状的维度相同。
    assert(n_blocks >= 0)
    assert (input_shape[0] == input_shape[1]) and (input_shape[1] == input_shape[2])
    assert (output_shape[0] == output_shape[1]) and (output_shape[1] == output_shape[2])

    #然后，定义了模型中使用到的一些变量，如输入形状input_shape、输入通道数input_nc和最小通道数minimal_nc。
    self.input_shape = input_shape
    self.input_nc = input_nc
    self.minimal_nc = 16

    #接下来，根据规范化层的类型来确定是否使用偏置项use_bias。
    # 然后，定义了激活函数activation
    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = norm_layer == nn.InstanceNorm3d

    #模型的构建通过一个列表model来完成。
    activation = nn.ReLU(True)


    #首先，添加了一个上采样块，使用build_upsampling_block方法构建。该块不进行上采样（up_sample=False），输入通道数和输出通道数均为input_nc，使用的是指定的规范化层、激活函数和上采样模式。
    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    # #upsample
    #接下来，根据输入形状和输出形状的比例关系，确定需要进行的最大上采样次数max_up。
    # 同时，计算出最大下采样次数max_down，并进行断言检查，确保上采样不会超出边界。
    max_up = int(np.max(np.log2(np.asarray(output_shape) / np.asarray(input_shape))))
    max_down = int(np.log2(input_nc/output_nc))
    assert max_up < max_down, 'upsampling overstep border'

    out_channel = input_nc

    #在每次循环中，根据上一层的输出通道数in_channel计算当前层的输出通道数out_channel
    for i in range(max_up-1):
      in_channel = out_channel

      #如果out_channel大于minimal_nc，则使用out_channel作为输出通道数；否则，使用minimal_nc作为输出通道数。
      out_channel = int(in_channel / 2) if (int(in_channel / 2) > self.minimal_nc) else self.minimal_nc
      
      #块的上采样标志up_sample设置为True，上采样模式为指定的模式，使用的规范化层、激活函数和是否进行上卷积均与前面的块相同。
      model += [
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    #添加一个最后的采样块，块的输入通道数为上一层的输出通道数in_channel，输出通道数为out_channel除以2（向下取整）。
    in_channel = out_channel
    out_channel = int(in_channel / 2)

    #块的上采样标志up_sample设置为True，上卷积标志up_conv设置为False，上采样模式为指定的模式，使用的规范化层、激活函数和是否进行上卷积均与前面的块相同。
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    #在模型的最后，添加一个3D卷积层nn.Conv3d，将输出通道数转换为output_nc。
    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
    ]

    # out activation
    #然后，添加一个输出激活函数out_activation
    model += [
      out_activation()
    ]

    #最后，使用nn.Sequential将列表model中的层组合成一个顺序的块，并将其作为结果保存在模型中。
    self.model = nn.Sequential(*model)

  #定义了build_upsampling_block函数，用于构建上采样块。
  #该函数接受一系列参数，
  # 包括输入通道数input_nc
  # 输出通道数output_nc
  # 规范化层norm_layer
  # 激活函数activation
  # 上采样标志up_sample
  # 上卷积标志up_conv
  # 上采样模式upsample_mode
  # 残差块数量block_n
  # 是否使用偏置项use_bias。
  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True, upsample_mode='nearest', block_n=3, use_bias=True):
    
    #首先创建一个空的列表blocks，用于存储构建的块。
    blocks = []

    #如果up_sample为True，则添加一个上采样和卷积块。
    #这里使用了自定义的Upsample_3DUnit类，它实现了3D空间的上采样和卷积操作。
    if up_sample:
      # #upsample and convolution

      #块中先进行上采样操作，将输入特征的尺寸增加一倍，然后使用卷积层将通道数从input_nc变为output_nc，并应用规范化层和激活函数。
      blocks += [
        Upsample_3DUnit(3, input_nc, output_nc, norm_layer, scale_factor=2, upsample_mode=upsample_mode, activation=activation, use_bias=use_bias)
      ]
    
    #如果up_conv为True，则根据block_n的值，使用循环构建多个残差块。
    # 每个残差块由一个Resnet_3DBlock模块和一个激活函数组成。
    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_3DBlock(output_nc, activation=activation, norm_layer=norm_layer, use_bias=use_bias),
          activation
        ]

    #最后，使用nn.Sequential将列表blocks中的层组合成一个顺序的块，并将其作为结果返回。
    return nn.Sequential(*blocks)

  #另外，该类还定义了前向传播方法forward，它将输入特征传递给模型进行计算，并返回计算结果。
  def forward(self, input):
    # #upsample and convolution
    return self.model(input)


#定义了Generator_3DNormResNetDecoder类，它是一个3D生成器模型的解码器部分。
class Generator_3DNormResNetDecoder(nn.Module):

  #构造函数__init__接受一系列参数，
  # 包括输入形状input_shape、
  # 输入通道数input_nc、
  # 输出形状output_shape、
  # 输出通道数output_nc、
  # 规范化层norm_layer、
  # 上采样模式upsample_mode、
  # 残差块数量n_blocks
  # 输出激活函数out_activation。
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DNormResNetDecoder, self).__init__()

    #在构造函数内部，首先进行断言检查，确保n_blocks为非负数，并且输入和输出形状的各维度大小相等。
    assert(n_blocks >= 0)
    assert (input_shape[0] == input_shape[1]) and (input_shape[1] == input_shape[2])
    assert (output_shape[0] == output_shape[1]) and (output_shape[1] == output_shape[2])

    #然后定义一些实例变量，包括输入形状input_shape和输入通道数input_nc
    self.input_shape = input_shape
    self.input_nc = input_nc

    #根据规范化层norm_layer的类型，确定是否使用偏置项use_bias
    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = norm_layer == nn.InstanceNorm3d

    #定义激活函数activation为ReLU激活函数
    activation = nn.ReLU(True)

    #接下来构建模型，首先添加一个上采样块，使用build_upsampling_block函数构建。
    # 该块的输入通道数和输出通道数都为input_nc，上采样标志为False，即不进行上采样操作，而是使用卷积操作。
    # 使用的上采样模式为upsample_mode，使用的块数量为n_blocks
    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    # #upsample
    #然后根据输入和输出形状的比例关系确定需要进行的上采样次数。
    # 通过计算输入形状和输出形状的对数差值，得到最大上采样次数max_up
    max_up = int(np.max(np.log2(np.asarray(output_shape) / np.asarray(input_shape))))
    max_down = int(np.log2(input_nc/output_nc))
    assert max_up < max_down, 'upsampling overstep border'

    out_channel = input_nc

    #在循环中，根据上一层的输出通道数和下一层的输入通道数计算出下一层的输出通道数。
    for i in range(max_up-1):
      in_channel = out_channel
      out_channel = int(in_channel / 2)

      #添加一个上采样块，使用build_upsampling_block函数构建。
      # 该块的输入通道数为上一层的输出通道数，输出通道数为计算得到的下一层的输出通道数。
      # 上采样标志为True，即进行上采样操作。
      # 使用的上采样模式为upsample_mode，使用的块数量为n_blocks。
      model += [
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)

    #最后一个上采样块与前面的块略有不同，它的up_conv参数为False，表示不使用上卷积。其余参数与前面的块相同
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    #在模型的最后，添加一个卷积层，将最后一层的输出通道数转换为output_nc，并使用规范化层对输出进行规范化。
    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
      norm_layer(output_nc)
    ]

    # out activation
    #添加输出激活函数out_activation
    model += [
      out_activation()
    ]

    #整个模型由一个序列块nn.Sequential组成，包含了构建的所有层。
    self.model = nn.Sequential(*model)


  #该函数接受一系列参数，
  # 包括输入通道数input_nc，
  # 输出通道数output_nc，
  # 规范化层norm_layer，
  # 激活函数activation，
  # 上采样标志up_sample，
  # 上卷积标志up_conv，
  # 上采样模式upsample_mode，
  # 块数量block_n
  # 使用偏置项的标志use_bias
  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    
    #内部定义了一个空的blocks列表，用于存储构建的层。
    blocks = []
    if up_sample:
      # #upsample and convolution

      #如果up_sample为True，表示进行上采样操作，则添加上采样层nn.Upsample，使用指定的上采样模式upsample_mode进行2倍的上采样。
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      #接着添加一个卷积层nn.Conv3d，将输入通道数input_nc转换为输出通道数output_nc，设置卷积核大小为3，填充为1，并使用偏置项。
      # 然后添加规范化层norm_layer对输出进行规范化，以及激活函数activation对输出进行激活。
      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    #如果up_conv为True，表示进行上卷积操作，则在循环中添加多个Resnet_3DBlock块，使用指定的激活函数activation、规范化层norm_layer和偏置项的标志use_bias进行构建
    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_3DBlock(output_nc, activation=activation, norm_layer=norm_layer, use_bias=use_bias),
          activation
        ]
    #最后，通过nn.Sequential将所有构建的层封装为一个序列块，并将其作为结果返回。
    return nn.Sequential(*blocks)
  
  #函数还定义了前向传播方法forward，将输入input传递给模型进行计算，并返回计算结果。
  def forward(self, input):
    # #upsample and convolution
    return self.model(input)


'''
3DGenerator_Decoder
=> Decoder + out_activation
'''
class Generator_3DLinearResNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DLinearResNetDecoder, self).__init__()
    assert(n_blocks >= 0)
    assert (input_shape[0] == input_shape[1]) and (input_shape[1] == input_shape[2])
    assert (output_shape[0] == output_shape[1]) and (output_shape[1] == output_shape[2])

    self.input_shape = input_shape
    self.input_nc = input_nc

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = norm_layer == nn.InstanceNorm3d

    activation = nn.ReLU(True)

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    # #upsample
    max_up = int(np.max(np.log2(np.asarray(output_shape) / np.asarray(input_shape))))
    max_down = int(np.log2(input_nc/output_nc))
    assert max_up < max_down, 'upsampling overstep border'

    out_channel = input_nc
    for i in range(max_up-1):
      in_channel = out_channel
      out_channel = int(in_channel / 2)
      model += [
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_3DBlock(output_nc, activation=activation, norm_layer=norm_layer, use_bias=use_bias)
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)


'''
3DGenerator_Decoder
=> Decoder + out_activation
'''
class Generator_3DLinearSTExpand2ResNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DLinearSTExpand2ResNetDecoder, self).__init__()
    assert(n_blocks >= 0)
    assert (input_shape[0] == input_shape[1]) and (input_shape[1] == input_shape[2])
    assert (output_shape[0] == output_shape[1]) and (output_shape[1] == output_shape[2])

    self.input_shape = input_shape
    self.input_nc = input_nc

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = norm_layer == nn.InstanceNorm3d

    activation = nn.ReLU(True)

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    # #upsample
    max_up = int(np.max(np.log2(np.asarray(output_shape) / np.asarray(input_shape))))
    max_down = int(np.log2(input_nc/output_nc))
    assert max_up < max_down, 'upsampling overstep border'

    out_channel = input_nc
    for i in range(max_up-1):
      in_channel = out_channel
      out_channel = int(in_channel / 2)
      model += [
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_ST3DBlock(output_nc, 2, activation=activation, norm_layer=norm_layer, use_bias=use_bias)
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)

class Generator_3DLinearSTShink2ResNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DLinearSTShink2ResNetDecoder, self).__init__()
    assert(n_blocks >= 0)
    assert (input_shape[0] == input_shape[1]) and (input_shape[1] == input_shape[2])
    assert (output_shape[0] == output_shape[1]) and (output_shape[1] == output_shape[2])

    self.input_shape = input_shape
    self.input_nc = input_nc

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = norm_layer == nn.InstanceNorm3d

    activation = nn.ReLU(True)

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    # #upsample
    max_up = int(np.max(np.log2(np.asarray(output_shape) / np.asarray(input_shape))))
    max_down = int(np.log2(input_nc/output_nc))
    assert max_up < max_down, 'upsampling overstep border'

    out_channel = input_nc
    for i in range(max_up-1):
      in_channel = out_channel
      out_channel = int(in_channel / 2)
      model += [
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_ST3DBlock(output_nc, 0.5, activation=activation, norm_layer=norm_layer, use_bias=use_bias)
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)

class Generator_3DLinearSTResNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DLinearSTResNetDecoder, self).__init__()
    assert(n_blocks >= 0)
    assert (input_shape[0] == input_shape[1]) and (input_shape[1] == input_shape[2])
    assert (output_shape[0] == output_shape[1]) and (output_shape[1] == output_shape[2])

    self.input_shape = input_shape
    self.input_nc = input_nc

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = norm_layer == nn.InstanceNorm3d

    activation = nn.ReLU(True)

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    # #upsample
    max_up = int(np.max(np.log2(np.asarray(output_shape) / np.asarray(input_shape))))
    max_down = int(np.log2(input_nc/output_nc))
    assert max_up < max_down, 'upsampling overstep border'

    out_channel = input_nc
    for i in range(max_up-1):
      in_channel = out_channel
      out_channel = int(in_channel / 2)
      model += [
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_ST3DBlock(output_nc, 1, activation=activation, norm_layer=norm_layer, use_bias=use_bias)
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)

class Generator_3DSTResNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DSTResNetDecoder, self).__init__()
    assert(n_blocks >= 0)
    assert (input_shape[0] == input_shape[1]) and (input_shape[1] == input_shape[2])
    assert (output_shape[0] == output_shape[1]) and (output_shape[1] == output_shape[2])

    self.input_shape = input_shape
    self.input_nc = input_nc

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = norm_layer == nn.InstanceNorm3d

    activation = nn.ReLU(True)

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    # #upsample
    max_up = int(np.max(np.log2(np.asarray(output_shape) / np.asarray(input_shape))))
    max_down = int(np.log2(input_nc/output_nc))
    assert max_up < max_down, 'upsampling overstep border'

    out_channel = input_nc
    for i in range(max_up-1):
      in_channel = out_channel
      out_channel = int(in_channel / 2)
      model += [
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias)
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_ST3DBlock(output_nc, 1, activation=activation, norm_layer=norm_layer, use_bias=use_bias),
          activation
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)

class Generator_3DDenseNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DDenseNetDecoder, self).__init__()
    assert(n_blocks >= 0)
    assert (input_shape[0] == input_shape[1]) and (input_shape[1] == input_shape[2])
    assert (output_shape[0] == output_shape[1]) and (output_shape[1] == output_shape[2])

    self.input_shape = input_shape
    self.input_nc = input_nc

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = norm_layer == nn.InstanceNorm3d

    activation = nn.ReLU(True)

    model = []
    num_layers = 3
    num_input_channels = input_nc
    bn_size = 4
    growth_rate = 16

    model += [
      Dense_3DBlock(num_layers, num_input_channels, True, bn_size, growth_rate, norm_layer, activation, use_bias)
    ]
    num_input_channels = num_layers*growth_rate

    # #upsample
    max_up = int(np.max(np.log2(np.asarray(output_shape) / np.asarray(input_shape))))
    max_down = int(np.log2(input_nc/output_nc))
    assert max_up < max_down, 'upsampling overstep border'

    for i in range(max_up):
      model += [
        nn.Upsample(scale_factor=2, mode=upsample_mode),
        norm_layer(num_input_channels),
        activation,
        nn.Conv3d(num_input_channels, num_input_channels, kernel_size=3, padding=1, bias=use_bias)
      ]

      model += [
        Dense_3DBlock(num_layers, num_input_channels, True, bn_size, growth_rate, norm_layer, activation, use_bias)
      ]
      num_input_channels = growth_rate*num_layers

    # channel reduction
    model += [
      nn.Conv3d(num_input_channels, output_nc, kernel_size=3, padding=1, bias=use_bias)
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)