
from self_attention import *
from encoder_decoder_utils import *
from swin_attention import *

'''
x=torch.rand(64,128,128)
layer=Pos_att(dim=64,
                kernel_size=1,
                stride=1
        )
'''


y0=torch.rand(1,64,128,128)
y1=torch.rand(1,256,64,64)
y2=torch.rand(1,512,32,32)
y3=torch.rand(1,1024,16,16)
y4=torch.rand(1,1024,8,8)
y5=torch.rand(1,704,4,4)
x=[y0,y1,y2,y3,y4,y5]
#(1,64,128,128) (1,256,64,64) (1,512,32,32) (1,1024,16,16) (1,1024,8,8) (1,704,4,4)
'''
#layer=CIA()
# img_size=128 patch_size=4 in_chans=96 embed_dim=96 norm_layer=nn.LayerNorm
layer=PatchEmbed(
            img_size=128, patch_size=4, in_chans=64, embed_dim=64,
            norm_layer=nn.LayerNorm)
# torch.Size([1, 16384, 64])
'''
for i in range(6):
   print('i:',i)
   B,C,H,W=x[i].shape
   layer=SwinIR(img_size=H, patch_size=4, in_chans=C,
                 embed_dim=C, depths=[6, 6, 6, 6], num_heads=[8, 8, 8, 8],
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='pixelshuffledirect', resi_connection='1conv'
                )
   out=layer(x[i])
   print(out.shape)
   print(out)