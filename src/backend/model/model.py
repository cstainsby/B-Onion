"""model.py

This file houses all of the implemented parts of the transformer
model I will be using

Link to the site which includes code and explanations of how stuff works
in depth:
https://nlp.seas.harvard.edu/2018/04/03/attention.html#data-loading

Link to blog post on how to get started with pytorch tranformer implementation
https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

Pytorch model documentation:
https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
https://pytorch.org/tutorials/beginner/transformer_tutorial.html


Some Notes:
The encoder takes a descrete sequence, like a sentence, and converts
    it into a continuous representation. 
The Decoder takes this continuous representation and makes an output 
    sequence one element at a time. Each element has dependancies on
    all the elements that came before.

The encoder and decoder are both made up of stacks that identical layers

Attention is this context is a function that maps a query and a set of key
    value pairs to an output where query, keys, values, and output are all vectors

"""


import math
import os
import copy
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import dataset

class Transformer(nn.Module):
    def __init__(
            self,
            num_tokens: int,             
            dim_model: int,              # defines how many dimensions the model has
            em_dim: int,                 # defines the embedding dimension
            num_heads: int,
            num_encoder_layers: int,     # defines how many layers the encoder will use
            num_decoder_layers: int,     # defines how many layers the decoder will use
            dropout_p: float = 0.5       # dropout percentage
        ) -> None:
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        self.positional_encoder = PositionalEncoding(
            d_model=dim_model,
            dropout=dropout_p
        )

        #NOTE: might have to play around with dim_feedforward for both
        #       the encoder and decoder

        encoder_layer = TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dropout=dropout_p
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        self.encoder = nn.Embedding(
            num_embeddings=num_tokens,
            embedding_dim=em_dim
        )

        decoder_layer = TransformerDecoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dropout=dropout_p
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )
        self.decoder = nn.Embedding(
            num_embeddings=num_tokens,
            embedding_dim=em_dim
        )
    
    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     self.encoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: Tensor, src_mask: Tensor):
        src = self.encoder(src) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)

        out = self.transformer_encoder(src, src_mask)
        out = self.deco


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)