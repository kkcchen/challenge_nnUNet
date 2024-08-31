from typing import Union, Type, List, Tuple

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args)

class AttentionBlock(nn.Module):
    def __init__(self,
                 encoding_scale_size, # (H, W, D)
                 encoding_channels, # d
                 in_channels, # d_0
                 memory_size, # N
                 ):
        
        assert len(encoding_scale_size) == 2 or len(encoding_scale_size) == 3, "encoding_scale_size must be 2D or 3D"
        
        super().__init__()
        self.encoding_scale_size = encoding_scale_size
        self.encoding_channels = encoding_channels
        self.memory_size = memory_size

        self.proj_layer = nn.Conv3d(in_channels=in_channels, out_channels=encoding_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.flatten_layer = nn.Flatten(start_dim=2)

        self.query_mem = nn.Linear(encoding_channels, encoding_channels, bias=False)
        self.key_mem   = nn.Linear(encoding_channels, encoding_channels, bias=False)
        self.value_mem = nn.Linear(encoding_channels, encoding_channels, bias=False)

        self.key_unet = nn.Linear(encoding_channels, encoding_channels, bias=False)
        self.value_unet = nn.Linear(encoding_channels, encoding_channels, bias=False)

        self.attention = nn.MultiheadAttention(embed_dim=encoding_channels, num_heads=1, dropout=0.0, batch_first=True)

        self.final_mlp = nn.Sequential(
            nn.Linear(encoding_channels, encoding_channels),
            Permute(0, 2, 1),
            nn.BatchNorm1d(encoding_channels),
            Permute(0, 2, 1),
            nn.ReLU(),
            nn.Linear(encoding_channels, encoding_channels),
            Permute(0, 2, 1),
            nn.BatchNorm1d(encoding_channels),
            Permute(0, 2, 1),
            nn.ReLU()
        )

        self.interpolate_mode = 'trilinear' if len(self.encoding_scale_size) == 3 else 'nearest'

    def forward(self, 
                feature_map, # x_c
                memory, # x_m
                shared_position_embedding # x_pos
                ):
        batch_size = feature_map.shape[0]

        # interpolation
        feature_map = nn.functional.interpolate(feature_map, size=self.encoding_scale_size, mode=self.interpolate_mode, align_corners=True)

        # linear project onto the encoding channels
        feature_map = self.proj_layer(feature_map)

        # flatten the last layers
        feature_map = self.flatten_layer(feature_map)

        # transpose
        feature_map = torch.transpose(feature_map, -1, -2)

        # add shared position embedding
        feature_map = feature_map + shared_position_embedding

        # query, key, value for memory branch
        query_mem = self.query_mem(memory)
        key_mem = self.key_mem(memory)
        value_mem = self.value_mem(memory)

        # key and value for unet branch
        key_unet = self.key_unet(feature_map)
        value_unet = self.value_unet(feature_map)

        # concatenate key and value
        key = torch.cat((key_mem, key_unet), dim=-2)
        value = torch.cat((value_mem, value_unet), dim=-2)

        # attention
        # print(query_mem.shape, key.shape, value.shape)
        feature_map, _ = self.attention(query_mem, key, value, need_weights=False)

        # 2-layer MLP
        x = self.final_mlp(feature_map)

        return x





