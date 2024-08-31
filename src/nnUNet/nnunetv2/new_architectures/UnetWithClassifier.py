from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
# from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from nnunetv2.new_architectures.MyUnetDecoder import MyUNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from nnunetv2.new_architectures.AttentionBlock import AttentionBlock

class MaxLayer(nn.Module):
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Apply torch.max along the specified dimension
        max_values, _ = torch.max(x, dim=self.dim)
        return max_values

class UnetWithClassifier(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        assert n_stages == 6, "only support 6 stages for now, because layers which are used for attention are hardcoded"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = MyUNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
        

        # class_layers = []
        # for channels in reversed(features_per_stage[:-1]):
        #     class_layers.append(
        #         nn.Sequential(
        #             nn.Flatten(start_dim=2),
        #             MaxLayer(-1),
        #             nn.Flatten(),
        #             nn.Linear(channels, 128),
        #             nn.BatchNorm1d(128),
        #             nn.ReLU(),
        #             nn.Linear(128, 128),
        #             nn.BatchNorm1d(128),
        #             nn.ReLU()
        #         )
        #     )
        
        # self.class_layers = nn.ModuleList(class_layers)

        # self.final_classifier = nn.Sequential(
        #     nn.Linear(128 * (len(features_per_stage) - 1), 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 3)
        # )
        n = 200
        d = 320
        hwd = (5, 8, 5)

        self.positional_embedding = nn.Parameter(torch.randn(n, d))
        self.global_memory = nn.Parameter(torch.randn(1, n, d))

        features_per_attention_stage = [features_per_stage[2], features_per_stage[4], features_per_stage[-2], features_per_stage[-5]]

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(encoding_scale_size=hwd,
                           encoding_channels=d,
                           in_channels=channels, 
                           memory_size=n
                           )
            for channels in features_per_attention_stage
        ])

        self.final_classifier = nn.Sequential(
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Linear(d, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        skips = self.encoder(x)
        decoded, intermediates = self.decoder(skips)
        
        # classification_outputs = []
        # for intermediate, class_layer in zip(intermediates, self.class_layers):
        #     classification_outputs.append(class_layer(intermediate))
        
        # classification_output_int = torch.cat(classification_outputs, dim=1)
        # classification_output = self.final_classifier(classification_output_int)

        unet_layers = [skips[2], skips[4], intermediates[0], intermediates[3]]
        memory = self.global_memory.repeat(x.shape[0], 1, 1)

        for unet_layer, attention_block in zip(unet_layers, self.attention_blocks):
            memory = attention_block(unet_layer, memory, self.positional_embedding)

        # memory should be batch by n by d, 200 by 320

        # mean pooling along the n dimension
        attention_output = memory.mean(dim=-2)

        # final classification
        classification_output = self.final_classifier(attention_output)

        return decoded, classification_output
    
    def freeze_encoder(self, freeze):
        print("Freezing encoder", freeze)
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def freeze_decoder(self, freeze):
        print("Freezing decoder", freeze)
        for param in self.decoder.parameters():
            param.requires_grad = not freeze
    
    def freeze_classifier(self, freeze):
        print("Freezing classifier", freeze)
        self.positional_embedding.requires_grad = not freeze
        self.global_memory.requires_grad = not freeze
        for attention_block in self.attention_blocks:
            for param in attention_block.parameters():
                param.requires_grad = not freeze
        for param in self.final_classifier.parameters():
            param.requires_grad = not freeze


    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
