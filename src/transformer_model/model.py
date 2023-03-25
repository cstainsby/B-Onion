import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class TransformerModel(nn.Module):
    """The defined structure for the text generating transformer
        Uses pytorch 

    Includes both an 
        - Encoder

    """
    def __init__(self, 
                 embedding_dim, 
                 hidden_dim, 
                 num_layers, 
                 num_heads, 
                 max_len, 
                 dropout_p = 0.2, 
                 embedding_matrix = None, 
                 num_embeddings=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_len = max_len
        self.dropout_p = dropout_p
        
        if embedding_matrix != None:
          self.encoder_embedding = nn.Embedding.from_pretrained(embedding_matrix).to(torch.float) # nn.Embedding(vocab_len, embedding_dim)
        else:
          self.encoder_embedding = nn.Embedding(
              num_embeddings=num_embeddings,
              embedding_dim=embedding_dim
          ).to(torch.float)
        
        # encoder definition
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len, dropout_p)
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout_p)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    
    def forward(self, src):
        out = self.encoder_embedding(src) * math.sqrt(self.embedding_dim)
        out = self.pos_encoder(out)

        # encoder output saved so that it can be used in decoder
        out = self.transformer_encoder(out)

        return out



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout_p=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)