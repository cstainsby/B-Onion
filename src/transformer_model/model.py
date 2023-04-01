import math
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class TransformerModel(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 d_model, 
                 nhead, 
                 num_layers,
                 max_seq_len,
                 dropout=0.1,
                 embedding_matrix = None, 
                 num_embeddings = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # allow loading in pretrained embeddings 
        # this will be useful for using glove pretrained embeddings
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=d_model
            )

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

        # print("Target size:", tgt.size(0))
        # print("target in forward:", tgt)
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(self, tgt.size(0)).to(tgt.device)

        batch_size = tgt.size(1)
        mock_memory = torch.zeros((self.num_layers, batch_size, self.d_model))

        out = self.transformer_decoder(tgt, memory=mock_memory) # tgt_mask=tgt_mask

        out = self.out(out)
        out = self.softmax(out)

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