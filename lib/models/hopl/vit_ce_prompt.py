import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import copy
from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from HOPL.lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_prompt

_logger = logging.getLogger(__name__)


class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output

class pix_module(nn.Module):
    '''
    多特征融合 AFF
    '''
    def __init__(self, in_dim1=16, in_dim2=8,dropout=0.1):
        super(pix_module, self).__init__()
        self.proj_q1 = nn.Linear(in_dim1, in_dim1, bias=False)
        self.proj_k1 = nn.Linear(in_dim1, in_dim1, bias=False)
        self.proj_v1 = nn.Linear(in_dim1, in_dim1, bias=False)
        self.proj_q2 = nn.Linear(in_dim2, in_dim2, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, in_dim2, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, in_dim2, bias=False)
        self.dropout1 = nn.Dropout(dropout)
    def forward(self, x1, x2):
        a,b,c,d=x1.size()
        if c==16:
            q = self.proj_q1(x2).flatten(2)
            k = self.proj_k1(x1).flatten(2)
            v = self.proj_v1(x1).flatten(2)
        else:
            q = self.proj_q2(x2).flatten(2)
            k = self.proj_k2(x1).flatten(2)
            v = self.proj_v2(x1).flatten(2)
        out=torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(torch.tensor(k.size(-1)).float())
        out=out.to(torch.float32)
        atten_weights=F.softmax(out,dim=-1)
        atten_out=torch.matmul(atten_weights,v)
        atten_out=atten_out.view(a,b,c,d)
        x = x1 + self.dropout1(atten_out)
        return x

class prompt_enhance(nn.Module):
    def __init__(self, in_planes=8, ratio=2):

        super(prompt_enhance, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        """
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        """
        # 利用1x1卷积代替全连接
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.relu1(self.avg_pool(x))
        max_out = self.relu1(self.max_pool(x))
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False,in_planes=8, ratio=2):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.PEM = prompt_enhance(in_planes, ratio)
        self.fovea = Fovea(smooth=smooth)
        self.cross_attention=pix_module()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, x):
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/2), :, :].contiguous()
        x1 = x[:, int(C / 2):, :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = self.conv0_1(x1)
        x1 = self.cross_attention(x0, x1)
        x1 = x1*self.PEM(x1)
        x0 = x0 + x1
        x0 = self.conv1x1(x0)
        return x0



class Fusion_MLP(nn.Module):
    def __init__(self, in_dim1, in_dim2,indim=192,dropout=0.1,d_model=245760):
        super(Fusion_MLP, self).__init__()
        self.proj_q1 = nn.Linear(in_dim1, in_dim2, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, in_dim2, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, in_dim2, bias=False)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.linear11 = nn.Linear(in_dim2, indim)
        self.linear12 = nn.Linear(indim, in_dim2)
        self.dropout1 = nn.Dropout(dropout)


    def forward(self, x1, x2, mask=None):
        q1 = self.proj_q1(x1)
        k2 = self.proj_k2(x2)
        v2 = self.proj_v2(x2)
        attn = torch.matmul(q1, k2.transpose(1, 2))/ torch.sqrt(torch.tensor(k2.size(-1)).float())
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2)
        x2 = x2 + self.dropout12(output)
        re1=x2.view(x2.size(0),-1)
        re1=self.norm12(re1)
        x2=re1.view(x2.size())
        src12 = self.linear12(self.dropout1(F.relu(self.linear11(x2))))
        x2 = x2 + self.dropout13(src12)
        re2 = x2.view(x2.size(0), -1)
        re2= self.norm13(re2)
        x2 = re2.view(x2.size())
        return x2,attn



class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # self.Fusion_MLP2=Fusion_MLP(in_dim1=768*2,in_dim2=768)
        self.part_BIF= Fusion_MLP(in_dim1=768*2, in_dim2=768)
        #self.fusion=ChannalAttention()

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_prompt = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #self.SelfAttentionModule=SelfAttentionModule(in_channels=3, out_channels=3)

        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # it's redundant
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        prompt parameters
        '''
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.prompt_type = prompt_type
        # various architecture
        if self.prompt_type in ['vipt_shaw', 'vipt_deep']:
            prompt_blocks = []
            block_nums = depth if self.prompt_type == 'vipt_deep' else 1
            for i in range(block_nums):
                prompt_blocks.append(Prompt_block(inplanes=embed_dim, hide_channel=8, smooth=True))
            self.prompt_blocks = nn.Sequential(*prompt_blocks)
            prompt_norms = []
            for i in range(block_nums):
                prompt_norms.append(norm_layer(embed_dim))
            self.prompt_norms = nn.Sequential(*prompt_norms)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb_img
        x_rgb1 = x[:, :3, :, :]
        z_rgb1 = z[:, :3, :, :]
        x_rgb2 = x[:, 3:6, :, :]
        z_rgb2 = z[:, 3:6, :, :]
        # depth thermal event images
        x_dte1 = x[:, 6:9, :, :]
        z_dte1 = z[:, 6:9, :, :]
        x_dte2 = x[:, 9:12, :, :]
        z_dte2 = z[:, 9:12, :, :]
        # overwrite x & z
        x1, z1 = x_rgb1, z_rgb1
        x2, z2 = x_rgb2, z_rgb2
        z1 = self.patch_embed(z1)
        x1 = self.patch_embed(x1)
        z_dte1 = self.patch_embed_prompt(z_dte1)
        x_dte1 = self.patch_embed_prompt(x_dte1)

        z2 = self.patch_embed_prompt(z2)
        x2 = self.patch_embed_prompt(x2)
        z_dte2 = self.patch_embed_prompt(z_dte2)
        x_dte2 = self.patch_embed_prompt(x_dte2)


        '''input prompt: by adding to rgb tokens'''
        if self.prompt_type in ['vipt_shaw', 'vipt_deep']:
            z_feat1 = token2feature(self.prompt_norms[0](z1))
            x_feat1 = token2feature(self.prompt_norms[0](x1))
            z_dte_feat1 = token2feature(self.prompt_norms[0](z_dte1))
            x_dte_feat1 = token2feature(self.prompt_norms[0](x_dte1))
            z_feat2 = token2feature(self.prompt_norms[0](z2))
            x_feat2 = token2feature(self.prompt_norms[0](x2))
            z_dte_feat2 = token2feature(self.prompt_norms[0](z_dte2))
            x_dte_feat2 = token2feature(self.prompt_norms[0](x_dte2))

            z_feat1 = torch.cat([z_feat1, z_dte_feat1], dim=1)
            x_feat1 = torch.cat([x_feat1, x_dte_feat1], dim=1)

            z_feat2 = torch.cat([z_feat2, z_dte_feat2], dim=1)
            x_feat2 = torch.cat([x_feat2, x_dte_feat2], dim=1)

            z_feat1 = self.prompt_blocks[0](z_feat1)
            x_feat1 = self.prompt_blocks[0](x_feat1)

            z_feat2 = self.prompt_blocks[0](z_feat2)
            x_feat2 = self.prompt_blocks[0](x_feat2)
            z_dte1 = feature2token(z_feat1)
            x_dte1 = feature2token(x_feat1)
            z_dte2 = feature2token(z_feat2)
            x_dte2 = feature2token(x_feat2)
            z_prompted1, x_prompted1= z_dte1, x_dte1
            z_prompted2, x_prompted2 = z_dte2, x_dte2

            z1 = z1 + z_dte1
            x1 = x1 + x_dte1
            z2 = z2 + z_dte2
            x2 = x2 + x_dte2
        else:
            z1 = z1 + z_dte1
            x1 = x1 + x_dte1
            z2 = z2 + z_dte2
            x2 = x2 + x_dte2


        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z1 += self.pos_embed_z
        x1 += self.pos_embed_x

        z2 += self.pos_embed_z
        x2 += self.pos_embed_x

        if self.add_sep_seg:
            x1 += self.search_segment_pos_embed
            z1 += self.template_segment_pos_embed
            x2 += self.search_segment_pos_embed
            z2 += self.template_segment_pos_embed


        x1 = combine_tokens(z1, x1, mode=self.cat_mode)
        x2 = combine_tokens(z2, x2, mode=self.cat_mode)
        if self.add_cls_token:
            x1 = torch.cat([cls_tokens, x1], dim=1)
            x2 = torch.cat([cls_tokens, x2], dim=1)

        x1 = self.pos_drop(x1)

        lens_z1 = self.pos_embed_z.shape[1]
        lens_x1 = self.pos_embed_x.shape[1]

        global_index1_t = torch.linspace(0, lens_z1 - 1, lens_z1, dtype=torch.int64).to(x1.device)
        global_index1_t = global_index1_t.repeat(B, 1)

        global_index1_s = torch.linspace(0, lens_x1 - 1, lens_x1, dtype=torch.int64).to(x1.device)
        global_index1_s = global_index1_s.repeat(B, 1)

        x2 = self.pos_drop(x2)

        lens_z2 = self.pos_embed_z.shape[1]
        lens_x2 = self.pos_embed_x.shape[1]

        global_index2_t = torch.linspace(0, lens_z2 - 1, lens_z2, dtype=torch.int64).to(x2.device)
        global_index2_t = global_index2_t.repeat(B, 1)

        global_index2_s = torch.linspace(0, lens_x2 - 1, lens_x2, dtype=torch.int64).to(x2.device)
        global_index2_s = global_index2_s.repeat(B, 1)

        removed_indexes_s1 = []
        removed_indexes_s2 = []
        removed_flag = False
        for i, blk in enumerate(self.blocks):
            '''
            add parameters prompt from 1th layer
            '''
            if i >= 1:
                if self.prompt_type in ['vipt_deep']:
                    x_ori1 = x1
                    # recover x to go through prompt blocks
                    lens_z1_new = global_index1_t.shape[1]
                    lens_x1_new = global_index1_s.shape[1]
                    z1 = x1[:, :lens_z1_new]
                    x1 = x1[:, lens_z1_new:]
                    if removed_indexes_s1 and removed_indexes_s1[0] is not None:
                        removed_indexes_cat1 = torch.cat(removed_indexes_s1, dim=1)
                        pruned_lens_x1 = lens_x1 - lens_x1_new
                        pad_x1 = torch.zeros([B, pruned_lens_x1, x1.shape[2]], device=x1.device)
                        x1 = torch.cat([x1, pad_x1], dim=1)
                        index1_all = torch.cat([global_index1_s, removed_indexes_cat1], dim=1)
                        C = x1.shape[-1]
                        x1 = torch.zeros_like(x1).scatter_(dim=1, index=index1_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x1)
                    x1 = recover_tokens(x1, lens_z1_new, lens_x1, mode=self.cat_mode)
                    x1 = torch.cat([z1, x1], dim=1)
                    # prompt
                    x1 = self.prompt_norms[i - 1](x1)  # todo
                    z_tokens1 = x1[:, :lens_z1, :]
                    x_tokens1 = x1[:, lens_z1:, :]
                    z_feat1 = token2feature(z_tokens1)
                    x_feat1 = token2feature(x_tokens1)
                    z_prompted1= self.prompt_norms[i](z_prompted1)
                    x_prompted1 = self.prompt_norms[i](x_prompted1)
                    z_prompt_feat1 = token2feature(z_prompted1)
                    x_prompt_feat1 = token2feature(x_prompted1)
                    z_feat1 = torch.cat([z_feat1, z_prompt_feat1], dim=1)
                    x_feat1 = torch.cat([x_feat1, x_prompt_feat1], dim=1)
                    z_feat1 = self.prompt_blocks[i](z_feat1)
                    x_feat1 = self.prompt_blocks[i](x_feat1)
                    z1 = feature2token(z_feat1)
                    x1 = feature2token(x_feat1)
                    z_prompted1, x_prompted1 = z1, x1
                    x1 = combine_tokens(z1, x1, mode=self.cat_mode)
                    # re-conduct CE
                    x1 = x_ori1 + candidate_elimination_prompt(x1, global_index1_t.shape[1], global_index1_s)

                    x_ori2 = x2
                    # recover x to go through prompt blocks
                    lens_z2_new = global_index2_t.shape[1]
                    lens_x2_new = global_index2_s.shape[1]
                    z2 = x2[:, :lens_z2_new]
                    x2 = x2[:, lens_z2_new:]
                    if removed_indexes_s2 and removed_indexes_s2[0] is not None:
                        removed_indexes_cat2 = torch.cat(removed_indexes_s2, dim=1)
                        pruned_lens_x2 = lens_x2 - lens_x2_new
                        pad_x2 = torch.zeros([B, pruned_lens_x2, x2.shape[2]], device=x2.device)
                        x2 = torch.cat([x2, pad_x2], dim=1)
                        index2_all = torch.cat([global_index2_s, removed_indexes_cat2], dim=1)
                        C = x2.shape[-1]
                        x2 = torch.zeros_like(x2).scatter_(dim=1,
                                                         index=index2_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                                                         src=x2)
                    x2 = recover_tokens(x2, lens_z2_new, lens_x2, mode=self.cat_mode)
                    x2 = torch.cat([z2, x2], dim=1)
                    # prompt
                    x2 = self.prompt_norms[i - 1](x2)  # todo
                    z_tokens2 = x2[:, :lens_z2, :]
                    x_tokens2 = x2[:, lens_z2:, :]
                    z_feat2 = token2feature(z_tokens2)
                    x_feat2 = token2feature(x_tokens2)
                    z_prompted2 = self.prompt_norms[i](z_prompted2)
                    x_prompted2 = self.prompt_norms[i](x_prompted2)
                    z_prompt_feat2 = token2feature(z_prompted2)
                    x_prompt_feat2 = token2feature(x_prompted2)
                    z_feat2 = torch.cat([z_feat2, z_prompt_feat2], dim=1)
                    x_feat2 = torch.cat([x_feat2, x_prompt_feat2], dim=1)
                    z_feat2 = self.prompt_blocks[i](z_feat2)
                    x_feat2 = self.prompt_blocks[i](x_feat2)
                    z2 = feature2token(z_feat2)
                    x2 = feature2token(x_feat2)
                    z_prompted2, x_prompted2 = z2, x2
                    x2 = combine_tokens(z2, x2, mode=self.cat_mode)
                    # re-conduct CE
                    x2 = x_ori2 + candidate_elimination_prompt(x2, global_index2_t.shape[1], global_index2_s)

            x1, global_index1_t, global_index1_s, removed_index1_s, attn1 = \
                blk(x1, global_index1_t, global_index1_s, mask_x, ce_template_mask, ce_keep_rate)
            x2, global_index2_t, global_index2_s, removed_index2_s, attn2 = \
                blk(x2, global_index2_t, global_index2_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s1.append(removed_index1_s)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s2.append(removed_index2_s)

        x1 = self.norm(x1)
        lens_x1_new = global_index1_s.shape[1]
        lens_z1_new = global_index1_t.shape[1]
        z1 = x1[:, :lens_z1_new]
        x1 = x1[:, lens_z1_new:]

        x2 = self.norm(x2)
        lens_x2_new = global_index2_s.shape[1]
        lens_z2_new = global_index2_t.shape[1]
        z2 = x2[:, :lens_z2_new]
        x2 = x2[:, lens_z2_new:]

        if removed_indexes_s1 and removed_indexes_s1[0] is not None:
            removed_indexes_cat1 = torch.cat(removed_indexes_s1, dim=1)

            pruned_lens_x1 = lens_x1 - lens_x1_new
            pad_x1 = torch.zeros([B, pruned_lens_x1, x1.shape[2]], device=x1.device)
            x1 = torch.cat([x1, pad_x1], dim=1)
            index_all1 = torch.cat([global_index1_s, removed_indexes_cat1], dim=1)
            # recover original token order
            C = x1.shape[-1]
            x1 = torch.zeros_like(x1).scatter_(dim=1, index=index_all1.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x1)

        x1 = recover_tokens(x1, lens_z1_new, lens_x1, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x1 = torch.cat([z1, x1], dim=1)

        if removed_indexes_s2 and removed_indexes_s2[0] is not None:
            removed_indexes_cat2 = torch.cat(removed_indexes_s2, dim=1)

            pruned_lens_x2 = lens_x2 - lens_x2_new
            pad_x2 = torch.zeros([B, pruned_lens_x2, x2.shape[2]], device=x2.device)
            x2 = torch.cat([x2, pad_x2], dim=1)
            index_all2 = torch.cat([global_index2_s, removed_indexes_cat2], dim=1)
            # recover original token order
            C = x2.shape[-1]
            x2 = torch.zeros_like(x2).scatter_(dim=1, index=index_all2.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                                             src=x2)

        x2 = recover_tokens(x2, lens_z2_new, lens_x2, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x2= torch.cat([z2, x2], dim=1)
        x3=torch.cat([x1, x2], dim=2)
        x11,at1=self.part_BIF(x3,x1)
        x22,at2 =self.part_BIF(x3, x2)
        x=x11+x22
        aux_dict = {
            "attn1": attn1,
            "attn2": attn2,
            "removed_indexes_s1": removed_indexes_s1,
            "removed_indexes_s2": removed_indexes_s2, # used for visualization
        }
        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
