import math
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_len, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tgt, memory=None):
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.embedding.embedding_dim).float())
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)

        if memory is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self, tgt.size(0)).to(tgt.device)
            output = self.transformer_decoder(tgt, tgt_mask=tgt_mask)
        else:
            tgt = tgt.permute(1, 0, 2) # Convert (batch_size, seq_len, emb_dim) to (seq_len, batch_size, emb_dim)
            memory = memory.permute(1, 0, 2) # Convert (batch_size, seq_len, emb_dim) to (seq_len, batch_size, emb_dim)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self, tgt.size(0)).to(tgt.device)
            memory_mask = nn.Transformer.generate_square_subsequent_mask(self, memory.size(0)).to(tgt.device)
            output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            output = output.permute(1, 0, 2) # Convert (seq_len, batch_size, emb_dim) to (batch_size, seq_len, emb_dim)

        output = self.out(output)
        output = self.softmax(output)

        return output

# class TransformerModel(nn.Module):
#     """The defined structure for the text generating transformer
#         Uses pytorch 

#     Includes both an 
#         - Encoder

#     """
#     def __init__(self, 
#                  embedding_dim, 
#                  hidden_dim, 
#                  num_layers, 
#                  num_heads, 
#                  max_len, 
#                  vocab_size,
#                  dropout_p = 0.2, 
#                  embedding_matrix = None, 
#                  num_embeddings=None):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.max_len = max_len
#         self.vocab_size = vocab_size
#         self.dropout_p = dropout_p
        
#         if embedding_matrix is not None:
#             self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
#         else:
#             self.embedding = nn.Embedding(
#                 num_embeddings=num_embeddings,
#                 embedding_dim=embedding_dim
#             )
        
#         self.pos_encoder = PositionalEncoding(embedding_dim, max_len, dropout_p)
#         decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout_p)
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

#     def forward(self, input_seq):
#         src = self.embedding(input_seq) * math.sqrt(self.embedding_dim)
#         src = self.pos_encoder(src)

#         output = self.transformer_decoder(src)

#         return output


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