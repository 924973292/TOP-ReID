import torch.nn as nn
from ..backbones.vit_pytorch import Attention
from ..backbones.vit_pytorch import DropPath
from ..backbones.vit_pytorch import Mlp
from ..backbones.vit_pytorch import trunc_normal_


class ReUnit(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ReBlock(nn.Module):

    def __init__(self, dim, num_heads,depth=1, mode=0):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.mode = mode
        for i in range(self.depth):
            self.blocks.append(
                ReUnit(dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.,
                       attn_drop=0.,
                       drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Reconstruct(nn.Module):

    def __init__(self, dim, num_heads,depth=1):
        super().__init__()
        self.re1 = ReBlock(dim, num_heads, depth=depth)
        self.re2 = ReBlock(dim, num_heads, depth=depth)

    def forward(self, x):
        re1 = self.re1(x)
        re2 = self.re2(x)
        return re1, re2


class CRM(nn.Module):

    def __init__(self, dim, num_heads, depth=1, miss='nothing'):
        super().__init__()
        self.RGBRE = Reconstruct(dim, num_heads, depth=depth)
        self.NIRE = Reconstruct(dim, num_heads, depth=depth)
        self.TIRE = Reconstruct(dim, num_heads, depth=depth)
        print("CRM HERE!!!")
        self.miss = miss
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, ma, mb, mc):
        if self.training:
            RGB_NI, RGB_TI = self.RGBRE(ma)
            NI_RGB, NI_TI = self.NIRE(mb)
            TI_RGB, TI_NI = self.TIRE(mc)
            loss_rgb = nn.MSELoss()(RGB_NI, mb) + nn.MSELoss()(RGB_TI, mc)
            loss_ni = nn.MSELoss()(NI_RGB, ma) + nn.MSELoss()(NI_TI, mc)
            loss_ti = nn.MSELoss()(TI_RGB, ma) + nn.MSELoss()(TI_NI, mb)
            loss = loss_rgb + loss_ni + loss_ti
            return loss
        else:
            if self.miss == None:
                pass
            elif self.miss == 'r':
                NI_RGB, NI_TI = self.NIRE(mb)
                TI_RGB, TI_NI = self.TIRE(mc)
                return (NI_RGB + TI_RGB) / 2
            elif self.miss == "n":
                RGB_NI, RGB_TI = self.RGBRE(ma)
                TI_RGB, TI_NI = self.TIRE(mc)
                return (RGB_NI + TI_NI) / 2
            elif self.miss == 't':
                RGB_NI, RGB_TI = self.RGBRE(ma)
                NI_RGB, NI_TI = self.NIRE(mb)
                return (RGB_TI + NI_TI) / 2
            elif self.miss == 'rn':
                TI_RGB, TI_NI = self.TIRE(mc)
                return TI_RGB, TI_NI
            elif self.miss == 'rt':
                NI_RGB, NI_TI = self.NIRE(mb)
                return NI_RGB, NI_TI
            elif self.miss == 'nt':
                RGB_NI, RGB_TI = self.RGBRE(ma)
                return RGB_NI, RGB_TI
