
import torch
import functools
from encoder_decoder_utils import *
from double_attention_no_AdaIN import *
encoder_block_list = [6, 12, 24, 16, 6]
bn_size = 4
activation = nn.ReLU(True)
encoder_channel_list = [640]
n_downsampling = len(encoder_block_list)
num_input_channels = encoder_channel_list[0]
encoder_norm_layer=nn.BatchNorm2d
encoder_input_channels=1
growth_rate = 32
x=torch.rand(640,1,128,128)#B C H W
#64——256
#256——448
#448——640
#640——832

def UNetLike_DownStep5(input_shape, encoder_input_channels, decoder_output_channels, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out=False):
  # 64, 32, 16, 8, 4
  encoder_block_list = [6, 12, 24, 16, 6]
  decoder_block_list = [1, 2, 2, 2, 2, 0]
  growth_rate = 32
  encoder_channel_list = [64]
  decoder_channel_list = [16, 16, 32, 64, 128, 256]
  decoder_begin_size = input_shape // pow(2, len(encoder_block_list))
  return UNetLike_DenseDimensionNet(input_shape,encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out)



class UNetLike_DenseDimensionNet(nn.Module):
  def __init__(self,input_shape, encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer=nn.BatchNorm2d, decoder_norm_layer=nn.BatchNorm3d, upsample_mode='nearest', decoder_feature_out=False):
    super(UNetLike_DenseDimensionNet, self).__init__()
    self.input_shape = input_shape
    self.decoder_channel_list = decoder_channel_list
    self.decoder_block_list = decoder_block_list
    self.n_downsampling = len(encoder_block_list)
    self.decoder_begin_size = decoder_begin_size
    self.decoder_feature_out = decoder_feature_out
    activation = nn.ReLU(True)
    bn_size = 4
    ##############
    ##########
    lr_mlp=0.01
    enable_full_resolution=8
    mlp_ratio=4
    use_checkpoint=False
    qkv_bias=True
    qk_scale=None
    drop_rate=0
    attn_drop_rate=0
    ##############
    # Encoder
    ##############
    if type(encoder_norm_layer) == functools.partial:
      use_bias = encoder_norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = encoder_norm_layer == nn.InstanceNorm2d
    ##############
    #编码第一层
    ##############
    #encoder_input_channels=1
    #encoder_channel_list[0]=64
    encoder_layers0 = [
      nn.ReflectionPad2d(3),
      nn.Conv2d(encoder_input_channels, encoder_channel_list[0], kernel_size=7, padding=0, bias=use_bias),
      encoder_norm_layer(encoder_channel_list[0]),
      activation
    ]
    self.encoder_layer = nn.Sequential(*encoder_layers0)

    num_input_channels = encoder_channel_list[0]
    print(input_shape)
    for index, channel in enumerate(encoder_block_list):
      # pooling
      down_layers = [
        #############归一化层，使网络平稳，有利于网络的训练
        encoder_norm_layer(num_input_channels),
        #############激活函数，用于在线性提取中引入非线性
        activation,
        #num_input_channels=64
        #num_ouput_channels=64
        nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, stride=2, padding=1, bias=use_bias),
        #############通过步幅为2的操作减小特征图的尺寸，同时通道数保持不变
        StyleBasicLayer(dim=num_input_channels,
                    input_resolution=(input_shape,input_shape),
                    depth=2,
                    num_heads=4,
                    window_size=8,
                    out_dim=num_input_channels,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    upsample=BilinearUpsample,
                    use_checkpoint=use_checkpoint
        )
      ]
      down_layers += [
        #############这是一个 Python 中的列表拼接操作，encoder_block_list[index]决定了在密集块中使用多少个卷积层
        #############num_input_channels这是输入到密集块的通道数，也就是前一个层的输出通道数。
        #############growth_rate：生长率，指定了每个密集块中每个卷积层的输出通道数。32
        #encoder_block_list = [6, 12, 32, 32, 12]
        Dense_2DBlock(encoder_block_list[index], num_input_channels, bn_size, growth_rate, encoder_norm_layer, activation, use_bias),
      ]
      ################这一行更新了下一个密集块的输入通道数。它计算了下一个密集块的输入通道数，这是当前输入通道数与当前密集块中卷积层的输出通道数之和。
      num_input_channels = num_input_channels + encoder_block_list[index] * growth_rate
      #64
      #64+6*32=256
      #256+12*32=640
      #640+32*32=1664
      #1664+32*32=2688
      #2688+12*32=3072

      # feature maps are compressed into 1 after the lastest downsample layers
      if index == (self.n_downsampling-1):
        down_layers += [
          nn.AdaptiveAvgPool2d(1)
        ]
      else:
        num_out_channels = num_input_channels // 2
        down_layers += [
          encoder_norm_layer(num_input_channels),
          activation,
          nn.Conv2d(num_input_channels, num_out_channels, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]
        num_input_channels = num_out_channels
      encoder_channel_list.append(num_input_channels)
      setattr(self, 'encoder_layer' + str(index), nn.Sequential(*down_layers))

    ##############
    # Linker
    ##############
    if type(decoder_norm_layer) == functools.partial:
      use_bias = decoder_norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = decoder_norm_layer == nn.InstanceNorm3d

    # linker FC
    # apply fc to link 2d and 3d
    self.base_link = nn.Sequential(*[
      ##############64**3*256 
      nn.Linear(encoder_channel_list[-1], decoder_begin_size**3*decoder_channel_list[-1]),
      ##############是 PyTorch 中用于实现Dropout正则化的层。Dropout 是一种用于防止神经网络过拟合的正则化技术
      nn.Dropout(0.5),
      activation
    ])

    for index, channel in enumerate(encoder_channel_list[:-1]):
      in_channels = channel
      out_channels = decoder_channel_list[index]
      link_layers = [
        Dimension_UpsampleCutBlock(in_channels, out_channels, encoder_norm_layer, decoder_norm_layer, activation, use_bias)
      ]
      setattr(self, 'linker_layer' + str(index), nn.Sequential(*link_layers))

    ##############
    # Decoder
    ##############
    for index, channel in enumerate(decoder_channel_list[:-1]):
      out_channels = channel
      in_channels = decoder_channel_list[index+1]
      decoder_layers = []
      decoder_compress_layers = []
      if index != (len(decoder_channel_list) - 2):
        decoder_compress_layers += [
          nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=use_bias),
          decoder_norm_layer(in_channels),
          activation
        ]
        for _ in range(decoder_block_list[index+1]):
          decoder_layers += [
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=use_bias),
            decoder_norm_layer(in_channels),
            activation
          ]
      decoder_layers += [
        Upsample_3DUnit(3, in_channels, out_channels, decoder_norm_layer, scale_factor=2, upsample_mode=upsample_mode, activation=activation, use_bias=use_bias)
      ]
      # If decoder_feature_out is True, compressed feature after upsampling and concatenation
      # can be obtained.
      if decoder_feature_out:
        setattr(self, 'decoder_compress_layer' + str(index), nn.Sequential(*decoder_compress_layers))
        setattr(self, 'decoder_layer' + str(index), nn.Sequential(*decoder_layers))
      else:
        setattr(self, 'decoder_layer' + str(index), nn.Sequential(*(decoder_compress_layers+decoder_layers)))
    # last decode
    decoder_layers = []
    decoder_compress_layers = [
      nn.Conv3d(decoder_channel_list[0] * 2, decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
      decoder_norm_layer(decoder_channel_list[0]),
      activation
    ]
    for _ in range(decoder_block_list[0]):
      decoder_layers += [
        nn.Conv3d(decoder_channel_list[0], decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
        decoder_norm_layer(decoder_channel_list[0]),
        activation
      ]
    if decoder_feature_out:
      setattr(self, 'decoder_compress_layer' + str(-1), nn.Sequential(*decoder_compress_layers))
      setattr(self, 'decoder_layer' + str(-1), nn.Sequential(*decoder_layers))
    else:
      setattr(self, 'decoder_layer' + str(-1), nn.Sequential(*(decoder_compress_layers + decoder_layers)))

    self.decoder_layer = nn.Sequential(*[
      nn.Conv3d(decoder_channel_list[0], decoder_output_channels, kernel_size=7, padding=3, bias=use_bias),
      decoder_out_activation()
    ])
  #############神经网络模型的前向传播算法，用于网络的前向传播
  def forward(self, input):
    print('input.shape:',input.shape)
    encoder_feature = self.encoder_layer(input)
    next_input = encoder_feature
    ###########循环下采样操作
    for i in range(self.n_downsampling):
      ##########这一行代码根据当前循环的索引 i 动态地将一个名为 linker_layer 的层应用到 next_input 上，
      ##########并将结果存储在名为 feature_linker<i> 的属性中。这个操作似乎是为了在下采样的过程中将中间特征存储下来以供后面的上采样使用。
      setattr(self, 'feature_linker' + str(i), getattr(self, 'linker_layer' + str(i))(next_input))
      ##########通过编码器层将 next_input 进行前向传播，更新 next_input
      next_input = getattr(self, 'encoder_layer'+str(i))(next_input)
    ##########在下采样循环结束后，将 next_input 展平并通过 base_link 层传播。这个层似乎是用来将特征进一步转换或连接到其他部分。
    next_input = self.base_link(next_input.view(next_input.size(0), -1))
    ##########将 next_input 重新变形，以适应解码器部分的输入大小。
    next_input = next_input.view(next_input.size(0), self.decoder_channel_list[-1], self.decoder_begin_size, self.decoder_begin_size, self.decoder_begin_size)

    for i in range(self.n_downsampling - 1, -2, -1):
      ############如果 i 是最后一层（self.n_downsampling - 1），则应用解码器层和可能的压缩层。
      if i == (self.n_downsampling - 1):
        
        if self.decoder_feature_out:
          next_input = getattr(self, 'decoder_layer' + str(i))(getattr(self, 'decoder_compress_layer' + str(i))(next_input))
        else:
          next_input = getattr(self, 'decoder_layer' + str(i))(next_input)
      ##############如果 i 不是最后一层，将当前的 next_input 与前面保存的 feature_linker 连接起来，然后应用解码器层。
      else:
        if self.decoder_feature_out:
          next_input = getattr(self, 'decoder_layer' + str(i))(getattr(self, 'decoder_compress_layer' + str(i))(torch.cat((next_input, getattr(self, 'feature_linker'+str(i+1))), dim=1)))
        else:
          next_input = getattr(self, 'decoder_layer' + str(i))(torch.cat((next_input, getattr(self, 'feature_linker'+str(i+1))), dim=1))

    return self.decoder_layer(next_input)
  
def get_norm_layer(norm_type='instance'):
  if norm_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
  elif norm_type == 'batch3d':
    norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
  elif norm_type == 'instance':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
  elif norm_type == 'instance3d':
    norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)
  elif norm_type == 'none':
    norm_layer = None
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
  return norm_layer


encoder_input_shape=(128,128)
encoder_input_nc=1
encoder_norm=nn.InstanceNorm2d
decoder_norm_layer = get_norm_layer(norm_type='batch')
activation_layer = nn.ReLU
output_nc=256
view1Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True)



print(view1Model)