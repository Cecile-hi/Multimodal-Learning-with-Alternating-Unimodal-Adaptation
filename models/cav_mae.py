# -*- coding: utf-8 -*-
# @Time    : 3/11/23 4:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : cav_mae.py

import os
os.environ['TORCH_HOME'] = './pretrained_models'
import random
import torch
import torch.nn as nn
import numpy as np
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
# from pos_embed import get_2d_sincos_pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype= float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_h_size, grid_w_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_w_size, grid_h_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x

# the finetuned CAV-MAE model
class CAVMAEFT(nn.Module):
    def __init__(self, label_dim, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=True):
        super().__init__()
        timm.models.vision_transformer.Block = Block
        print('Use norm_pix_loss: ', norm_pix_loss)

        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(12 - modality_specific_depth)])

        self.norm_a = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        # self.norm = norm_layer(embed_dim)

        # self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, label_dim))

        self.initialize_weights()

        print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

    def get_patch_num(self, input_shape, stride):
        test_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        test_proj = torch.nn.Conv2d(1, 4, kernel_size=(16, 16), stride=(stride, stride))
        test_output = test_proj(test_input)
        print(test_output.shape)
        return test_output.shape[2], test_output[3], test_output[2] * test_output[2]

    def initialize_weights(self):
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, a, v, mode):
        # multi-modal fine-tuning, our default method for fine-tuning
        if mode == 'multimodal':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_v:
                v = blk(v)

            x = torch.cat((a, v), dim=1)

            for blk in self.blocks_u:
                x = blk(x)
            x = self.norm(x)

            x = x.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # finetune with only audio (and inference with only audio when the model is finetuned with only audio)
        elif mode == 'audioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # note here uses the 'a' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)
            x = a.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # finetune with only image (and inference with only audio when the model is finetuned with only image)
        elif mode == 'videoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)

            # note here uses the 'v' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_v(v)
            x = v.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # used in case that the model is finetuned with both modality, but in inference only audio is given
        elif mode == 'missingaudioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = a
            for blk in self.blocks_u:
                u = blk(u) # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                a = blk(a, 'a') # note here use modality-specific normalization
            a = self.norm_a(a)
            a = a.mean(dim=1)

            # average the output of the two forward passes
            x = (u + a) / 2
            x = self.mlp_head(x)
            return x

        # used in case that the model is fine-tuned with both modality, but in inference only image is given
        elif mode == 'missingvideoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = v
            for blk in self.blocks_u:
                u = blk(u) # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                v = blk(v, 'v') # note here use modality-specific normalization
            v = self.norm_v(v)
            v = v.mean(dim=1)

            # average the output of the two forward passes
            x = (u + v) / 2
            x = self.mlp_head(x)
            return x

    # for retrieval
    def forward_feat(self, a, v, mode='av'):
        # return both audio and visual
        if mode == 'av':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_v:
                v = blk(v)

            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)

            for blk in self.blocks_u:
                v = blk(v, 'v')

            v = self.norm_v(v)
            return a, v

        # return only audio
        elif mode == 'a':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_u:
                a = blk(a, 'a')

            a = self.norm_a(a)
            return a
        elif mode == 'v':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)

            for blk in self.blocks_u:
                v = blk(v, 'v')

            v = self.norm_v(v)
            return v
