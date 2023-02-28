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
import time
import numpy as np
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, LayerNorm
from torch.utils.data import dataset



class LanguageModel(nn.Module):
    """A Wrapper for the transformer used by our model, 
    
    Includes all the functionality needed for intereact with the model in the app
    """
    def __init__(self,
                
                # hyper parameters
                num_tokens: int,             
                dim_model: int,                     # defines how many dimensions the model has (maximum sequence length)
                em_dim: int,                        # defines the embedding dimension
                num_heads: int,                     # the number of attention heads
                dimensions_of_feedforward: int,     # dimension of the feed forward layer
                num_encoder_layers: int,            # defines how many layers the encoder will use
                num_decoder_layers: int,            # defines how many layers the decoder will use
                dropout_p: float = 0.5,              # dropout percentage/rate) -> None:
        ) -> None:
        super().__init__()

        self.model = EncoderDecoder(
            num_tokens=num_tokens,
            dim_model=dim_model,
            em_dim=em_dim,
            num_heads=num_heads,
            dimensions_of_feedforward=dimensions_of_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout_p=dropout_p
        )
        
        self.generator = Generator()

        self.positional_encoder = PositionalEncoding(
            embedding_dimension=em_dim,
            dropout=dropout_p,
            max_sequence_length= dim_model
        )
    
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """Forward method of full transformer 
        
        Copied from implementation: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        with some modifications, no self.out, using Generator forward instead"""
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt)
        out = self.generator(transformer_out)

        return out


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0



class EncoderDecoder(nn.Module):
    def __init__(
            self,
            num_tokens: int,             
            dim_model: int,                     # defines how many dimensions the model has (maximum sequence length)
            em_dim: int,                        # defines the embedding dimension
            num_heads: int,                     # the number of attention heads
            dimensions_of_feedforward: int,     # dimension of the feed forward layer
            num_encoder_layers: int,            # defines how many layers the encoder will use
            num_decoder_layers: int,            # defines how many layers the decoder will use
            dropout_p: float = 0.5              # dropout percentage/rate
        ) -> None:
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        # -----------------------
        #   ENCODER 
        # -----------------------
        encoder_layer = TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dimensions_of_feedforward,
            dropout=dropout_p
        )
        encoder_norm = LayerNorm(dim_model)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm
        )
        self.encoder = nn.Embedding(
            num_embeddings=num_tokens,
            embedding_dim=em_dim
        )
        
        # -----------------------
        #   DECODER
        # -----------------------
        decoder_layer = TransformerDecoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward= dimensions_of_feedforward,
            dropout=dropout_p
        )
        decoder_norm = LayerNorm(dim_model)
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=decoder_norm
        )
        self.decoder = nn.Embedding(
            num_embeddings=num_tokens,
            embedding_dim=em_dim
        )
    
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        return output

class Generator(nn.Module):
    """
    The language model generator is a linear layer that maps the embedding dimension to the vocabulary size.
    """

    def __init__(self, embedding_dimension: int, number_of_tokens: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_tokens = number_of_tokens
        self.linear = nn.Linear(embedding_dimension, number_of_tokens)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the language model head.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        output dimensions are: (batch_size, sequence_length, number_of_tokens)
        """
        # Compute the linear layer
        # linear_output dimensions are: (batch_size, sequence_length, number_of_tokens)
        linear_output = self.linear(x)

        return F.softmax(linear_output, dim=-1)

    
class PositionalEncoding(nn.Module):
    """Implement the PE function.
    
    Allows the model to learn about the order of the input tokens"""
    def __init__(self, d_model: int, dropout: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    