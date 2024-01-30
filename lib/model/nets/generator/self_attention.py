import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features =  in_features
        hidden_features =  in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class conv1(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(conv1, self).__init__()
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        
    def forward(self, x):
        # 检查输入张量的设备
        device = x.device

        # 如果conv1x1的权重不在相同设备上，将其移动到相同的设备
        if self.conv1x1.weight.device != device:
            self.conv1x1 = self.conv1x1.to(device)

        x = self.conv1x1(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.mlp = Mlp(in_features=dim, act_layer=act_layer, drop=drop_ratio)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        #print('x.shape:',x.shape)
        B,C,H,W=x.shape
        x=x.reshape(-1,C)
        x=self.norm1(x.to(self.norm1.weight.device))
        x=x.reshape(1,-1,C)
        x = x+self.attn(x)
        x= x + self.mlp(self.norm2(x))
        x=x.reshape(B,C,H,W)

        #img=transforms.ToPILImage()(torch.clamp(x.squeeze(0).cpu(),0,1))
        #grid_img=torchvision.utils.make_grid(x.squeeze(0).cpu(),nrow=8,normalize=True)
        #plt.imshow(grid_img.permute(1,2,0)[:,:,32].cpu().detach().numpy(),cmap='gray',vmin=0,vmax=1)
        #plt.show()
        #print('x.shape:',x.shape)
        return x

class CIA(nn.Module):
    def __init__(self):
        super(CIA,self).__init__()

    def forward(self,x):
        B,C,H,W=x.shape
        x1_1=x.reshape(-1,C)

        x2_1=x.reshape(C,-1)
        x2_2=x2_1 @ x1_1
        x2_2=x2_2.softmax(dim=-1)

        x3_1=x.reshape(C,-1)

        x3_2=x2_2 @ x3_1
        x3_2=x3_2.softmax(dim=-1)
        x3_3=x3_2.reshape(B,C,H,W)

        x_f=x+x3_3.softmax(dim=-1)
        
        return x_f
    
class AIA_1(nn.Module):
    def __init__(self,dim):
        super(AIA_1,self).__init__()

        self.conv1x1 = nn.Conv2d(dim, dim//8, kernel_size=1, stride=1,padding=0)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=2,padding=1)
        #self.conv1x1_1 = nn.Conv2d(8, 8, kernel_size=1, stride=1,padding=0)
        #self.conv1x1_2 = nn.Conv2d(16, 32, kernel_size=1, stride=1,padding=0)
        #self.conv1x1_3 = nn.Conv2d(64, 128, kernel_size=1, stride=1,padding=0)

        self.relu=nn.ReLU(inplace=True)
        #这是Leaky ReLU的负斜率（slope），表示当输入值小于零时，小于零的部分会乘以0.2，以引入一个小的负梯度，以避免神经元完全关闭。
        #当设置为True时，Leaky ReLU将直接修改输入张量，而不是创建一个新的张量。
        self.leakyrelu=nn.LeakyReLU(0.2,True)
        self.sigm=nn.Sigmoid()
        self.batchnorm=nn.BatchNorm2d(dim)
        #self.fc1=nn.Linear(in_features=8*64*64,out_features=16*32*32)
        #self.fc2=nn.Linear(in_features=32*32*32,out_features=64*16*16)
        #self.fc3=nn.Linear(in_features=128*16*16,out_features=256*8*8)
        #这里的kernel_size参数表示平均池化的核大小，stride表示池化操作的步幅，padding是填充
        self.avgpool=nn.AvgPool2d(kernel_size=2,stride=2,padding=0)

    def forward(self,x):
        #x [1 64 128 128]
        #x=self.conv1x1(x)
        #x [1 8 128 128]
        #x=self.leakyrelu(x)
        #x [1 8 128 128]
        #(输入高度 - 池化核的高度 + 2 * 填充) / 步长 + 1
        #(输入宽度 - 池化核的宽度 + 2 * 填充) / 步长 + 1
        #通道数不变
        B,C,H,W=x.shape
        #layer=Pos_att(C).cuda()
        layer=CIA().cuda()
        x1_1=layer(x).cuda()
        #x1_1 [1 64 128 128]

        x1_2=self.conv3x3(x1_1.to(self.conv3x3.weight.device))
        x1_2=self.batchnorm(x1_2)
        x1_2=self.relu(x1_2)

        #x1_2 [1 64 64 64]
        x2_1=self.conv3x3(x.to(self.conv3x3.weight.device))
        x2_1=self.batchnorm(x2_1)
        x2_1=self.relu(x2_1)
        #x2_1 [1 64 64 64]

        x2_2=x1_2 @ x2_1.transpose(-2,-1)
        x2_2=x2_2.softmax(dim=-1)
        #x1_1 [1 64 64 64]

        x3_1=self.conv3x3(x)
        #x3_1 [1 64 64 64]
        x3_1=self.batchnorm(x3_1)
        x3_1=self.relu(x3_1)

        x4_1=self.conv3x3(x)
        #x4_1 [1 64 64 64]
        x4_1=self.batchnorm(x4_1)
        #print('x4_1.shape:',x4_1.shape)
        x4_2=self.leakyrelu(x4_1)
        #x4_2 [1 64 64 64]
        #print('x4_2.shape:',x4_2.shape)
        #print('x4_3.shape:',x4_3.shape)
        x4_3=self.sigm(x4_2)
        #x4_3 [1 64 64 64]
        #print('x4_4.shape:',x4_4.shape)
        x3_2=x3_1 @ x4_3.transpose(-2, -1)
        #x3_2 [1 64 64 64]
        #print('x3_2.shape:',x3_2.shape)
        x3_2=x3_2.softmax(dim=-1)
        x3_3=x1_2 @ x3_2.transpose(-2,-1)
        x3_3=x3_3.softmax(dim=-1)
        x_f1=x3_3+x2_2

        x_f2=self.conv3x3(x)
        x_f2=self.batchnorm(x_f2)
        x_f2=self.relu(x_f2)

        x_f=x_f1+x_f2
        x_f=self.relu(x_f)
        x_f=F.interpolate(x_f,size=(H,W),mode='bilinear',align_corners=False)
        #x_f [1 64 128 128]

        #grid_img=torchvision.utils.make_grid(x_f.squeeze(0).cpu(),nrow=8,normalize=True)
        #plt.imshow(grid_img.permute(1,2,0)[:,:,32].cpu().detach().numpy(),cmap='gray',vmin=0,vmax=1)
        #plt.show()
        return x_f

class Pos_att(nn.Module):
    def __init__(self, dim):
        super(Pos_att,self).__init__()

        self.conv_1 = nn.Conv2d(dim, dim//8, kernel_size=1, stride=1)

    def forward(self,x):
        #x [C H W]
        B,C,H,W=x.shape
        #x_1 [C//8 H W]
        x_1=self.conv_1(x)
        #print('x_1.shape:',x_1.shape)
        #x_2 [W H C//8]
        x_1=x_1.reshape(-1,H,W)
        x_2=x_1.permute(2,1,0).reshape(-1,C//8)
        #print('x_2.shape:',x_2.shape)
        #x_3 [C//8 W H]
        x_3=x_1.permute(0,2,1).reshape(-1,W*H)
        #print('x_3.shape:',x_3.shape)
        #x_3 [W H W H]
        x_4=x_2 @ x_3
        x_4=x_4.softmax(dim=-1)
        #print('x_4.shape:',x_4.shape)
        
        x_5=x.reshape(-1,W*H)
        #print('x_5.shape:',x_5.shape)
        x_6=x_5 @ x_4
        #print('x_6.shape:',x_6.shape)
        x_6=x_6.softmax(dim=-1)
        x_7=x_6.reshape(B,C,H,W)
        x_f=x_7+x
        x_f=x_f.softmax(dim=-1)
        
        
        return x_f
    
class AIA(nn.Module):
    def __init__(self,dim):
        super(AIA,self).__init__()

        self.conv1x1 = nn.Conv2d(dim, dim//8, kernel_size=1, stride=1,padding=0)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=2,padding=1)
        #self.conv1x1_1 = nn.Conv2d(8, 8, kernel_size=1, stride=1,padding=0)
        #self.conv1x1_2 = nn.Conv2d(16, 32, kernel_size=1, stride=1,padding=0)
        #self.conv1x1_3 = nn.Conv2d(64, 128, kernel_size=1, stride=1,padding=0)

        self.relu=nn.ReLU(inplace=True)
        #这是Leaky ReLU的负斜率（slope），表示当输入值小于零时，小于零的部分会乘以0.2，以引入一个小的负梯度，以避免神经元完全关闭。
        #当设置为True时，Leaky ReLU将直接修改输入张量，而不是创建一个新的张量。
        self.leakyrelu=nn.LeakyReLU(0.2,True)
        self.sigm=nn.Sigmoid()
        #self.fc1=nn.Linear(in_features=8*64*64,out_features=16*32*32)
        #self.fc2=nn.Linear(in_features=32*32*32,out_features=64*16*16)
        #self.fc3=nn.Linear(in_features=128*16*16,out_features=256*8*8)
        #这里的kernel_size参数表示平均池化的核大小，stride表示池化操作的步幅，padding是填充
        self.avgpool=nn.AvgPool2d(kernel_size=2,stride=2,padding=0)

    def forward(self,x):
        #x [1 64 128 128]
        #x=self.conv1x1(x)
        #x [1 8 128 128]
        #x=self.leakyrelu(x)
        #x [1 8 128 128]
        #(输入高度 - 池化核的高度 + 2 * 填充) / 步长 + 1
        #(输入宽度 - 池化核的宽度 + 2 * 填充) / 步长 + 1
        #通道数不变
        B,C,H,W=x.shape
        layer=Pos_att(C).cuda()
        x1_1=layer(x)
        #x1_1 [1 64 128 128]

        x1_2=self.conv3x3(x1_1)
        #x1_2 [1 64 64 64]
        x2_1=self.conv3x3(x)
        #x2_1 [1 64 64 64]
        x2_2=x1_2 @ x2_1.transpose(-2,-1)
        x2_2=x2_2.softmax(dim=-1)
        #x1_1 [1 64 64 64]

        x3_1=self.conv3x3(x)
        #x3_1 [1 64 64 64]

        x4_1=self.conv3x3(x)
        #x4_1 [1 64 64 64]
        #print('x4_1.shape:',x4_1.shape)
        x4_2=self.leakyrelu(x4_1)
        #x4_2 [1 64 64 64]
        #print('x4_2.shape:',x4_2.shape)
        #print('x4_3.shape:',x4_3.shape)
        x4_3=self.sigm(x4_2)
        #x4_3 [1 64 64 64]
        #print('x4_4.shape:',x4_4.shape)
        x3_2=x3_1 @ x4_3.transpose(-2, -1)
        #x3_2 [1 64 64 64]
        #print('x3_2.shape:',x3_2.shape)
        x3_2=x3_2.softmax(dim=-1)
        x3_3=x1_2 @ x3_2.transpose(-2,-1)
        x3_3=x3_3.softmax(dim=-1)
        x_f1=x3_3+x2_2
        x_f2=self.conv3x3(x)
        x_f=x_f1+x_f2
        x_f=self.relu(x_f)
        x_f=F.interpolate(x_f,size=(H,W),mode='bilinear',align_corners=False)
        #x_f [1 64 128 128]
        #grid_img=torchvision.utils.make_grid(x_f.squeeze(0).cpu(),nrow=8,normalize=True)
        #plt.imshow(grid_img.permute(1,2,0)[:,:,32].cpu().detach().numpy(),cmap='gray',vmin=0,vmax=1)
        #plt.show()
        return x_f
'''

class AIA(nn.Module):
    def __init__(self,dim):
        super(AIA,self).__init__()

        self.conv1x1 = nn.Conv2d(dim, dim//8, kernel_size=1, stride=1,padding=0)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=2,padding=1)
        #self.conv1x1_1 = nn.Conv2d(8, 8, kernel_size=1, stride=1,padding=0)
        #self.conv1x1_2 = nn.Conv2d(16, 32, kernel_size=1, stride=1,padding=0)
        #self.conv1x1_3 = nn.Conv2d(64, 128, kernel_size=1, stride=1,padding=0)

        self.relu=nn.ReLU(inplace=True)
        self.batchnorm=nn.BatchNorm2d(dim)
        #这是Leaky ReLU的负斜率（slope），表示当输入值小于零时，小于零的部分会乘以0.2，以引入一个小的负梯度，以避免神经元完全关闭。
        #当设置为True时，Leaky ReLU将直接修改输入张量，而不是创建一个新的张量。
        self.leakyrelu=nn.LeakyReLU(0.2,True)
        self.sigm=nn.Sigmoid()
        #self.fc1=nn.Linear(in_features=8*64*64,out_features=16*32*32)
        #self.fc2=nn.Linear(in_features=32*32*32,out_features=64*16*16)
        #self.fc3=nn.Linear(in_features=128*16*16,out_features=256*8*8)
        #这里的kernel_size参数表示平均池化的核大小，stride表示池化操作的步幅，padding是填充
        self.avgpool=nn.AvgPool2d(kernel_size=2,stride=2,padding=0)

    def forward(self,x):
        #x [1 64 128 128]
        #x=self.conv1x1(x)
        #x [1 8 128 128]
        #x=self.leakyrelu(x)
        #x [1 8 128 128]
        #(输入高度 - 池化核的高度 + 2 * 填充) / 步长 + 1
        #(输入宽度 - 池化核的宽度 + 2 * 填充) / 步长 + 1
        #通道数不变
        B,C,H,W=x.shape
        layer=Pos_att(C).cuda()
        #layer=CIA().cuda()
        x1_1=layer(x)
        #x1_1 [1 64 128 128]

        x1_2=self.conv3x3(x1_1.to(self.conv3x3.weight.device))
        x1_2=self.batchnorm(x1_2)
        x1_2=self.relu(x1_2)

        #x1_2 [1 64 64 64]
        x2_1=self.conv3x3(x.to(self.conv3x3.weight.device))
        x2_1=self.batchnorm(x2_1)
        x2_1=self.relu(x2_1)
        #x2_1 [1 64 64 64]

        x2_2=x1_2 @ x2_1.transpose(-2,-1)
        x2_2=x2_2.softmax(dim=-1)
        #x1_1 [1 64 64 64]

        x3_1=self.conv3x3(x)
        #x3_1 [1 64 64 64]
        x3_1=self.batchnorm(x3_1)
        x3_1=self.relu(x3_1)

        x4_1=self.conv3x3(x)
        #x4_1 [1 64 64 64]
        x4_1=self.batchnorm(x4_1)
        #print('x4_1.shape:',x4_1.shape)
        x4_2=self.leakyrelu(x4_1)
        #x4_2 [1 64 64 64]
        #print('x4_2.shape:',x4_2.shape)
        #print('x4_3.shape:',x4_3.shape)
        x4_3=self.sigm(x4_2)
        #x4_3 [1 64 64 64]
        #print('x4_4.shape:',x4_4.shape)
        x3_2=x3_1 @ x4_3.transpose(-2, -1)
        #x3_2 [1 64 64 64]
        #print('x3_2.shape:',x3_2.shape)
        x3_2=x3_2.softmax(dim=-1)
        x3_3=x1_2 @ x3_2.transpose(-2,-1)
        x3_3=x3_3.softmax(dim=-1)
        x_f1=x3_3+x2_2

        x_f2=self.conv3x3(x)
        x_f2=self.batchnorm(x_f2)
        x_f2=self.relu(x_f2)
        
        x_f=x_f1+x_f2
        x_f=self.relu(x_f)
        x_f=F.interpolate(x_f,size=(H,W),mode='bilinear',align_corners=False)
        #x_f [1 64 128 128]
        return x_f
'''    
'''

    
class AIA(nn.Module):
    def __init__(self):
        super(AIA,self).__init__()
    
    def forward(self,x):
        B,C,H,W=x.shape
        x=AIA_1(dim=C)(x).cuda()+AIA_2(dim=C)(x).cuda()
        return x
'''
#Conv and Transformer
class CAT(nn.Module):
    def __init__(self, dim):
        super(CAT,self).__init__()
        #64-64
        self.conv3x3_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1,padding=1)
        #32-32
        self.conv3x3_2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, stride=1,padding=1)
        self.conv1x1_1 = nn.Conv2d(dim, dim//2, kernel_size=1, stride=1,padding=0)
        self.conv1x1_2 = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1,padding=0)

        self.Trans=Block(dim//2,num_heads=2)
        self.relu=nn.ReLU(inplace=True)
        self.Tanh=nn.Tanh()
        self.sigm=nn.Sigmoid()
        self.batchnorm_1=nn.BatchNorm2d(dim)
        self.batchnorm_2=nn.BatchNorm2d(dim//2)

    def forward(self,x):
        B,C,H,W=x.shape
        #[1,64,128,128]
        x1_1=self.conv3x3_1(x)
        x1_1=self.batchnorm_1(x1_1)
        x1_1=self.relu(x1_1)
        #print('x1_1.shape',x1_1.shape)
        #[1,64,128,128]
        x1_2=self.conv1x1_1(x1_1)
        x1_2=self.batchnorm_2(x1_2)
        x1_2=self.relu(x1_2)
        #[1,32,128,128]

        #上分枝
        x1_3=self.conv3x3_2(x1_2)
        x1_3=self.batchnorm_2(x1_3)
        x1_3=self.relu(x1_3)
        #[1,32,128,128]

        #下分枝
        x1_4=self.Trans(x1_2)
        #[1,32,128,128]

        x1=torch.cat((x1_3, x1_4), dim=1)
        #[1,64,128,128]

        x1_4=self.conv3x3_1(x1)
        x1_4=self.conv1x1_1(x1_4)
        x1_4=self.batchnorm_2(x1_4)
        x1_4=self.relu(x1_4)

        #[1,32,128,128]

        x1_5=torch.cat((x1_2, x1_4), dim=1)
        #[1,64,128,128]

        x1_6=self.conv3x3_1(x1_5)
        x1_6=self.batchnorm_1(x1_6)
        x1_6=self.relu(x1_6)
        #[1,64,128,128]

        x1_7=torch.cat((x1_1, x1_6), dim=1)
        #[1,128,128,128]
        x1_f=self.relu(x1_7)
        x1_f=self.conv1x1_2(x1_f)
        x1_f=self.sigm(x1_f)

        #[1,64,128,128]
        x=x+x1_f
        return x
        