# https://www.kaggle.com/code/mlwhiz/textcnn-pytorch-and-kerasv

import numpy as np
import torch
import torch.nn as nn
from torch.functional import F

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')

class CNN_Text(nn.Module):
    
    def __init__(self):
        super(CNN_Text, self).__init__()

        embed_size = 300 # how big is each word vector
        max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
        maxlen = 70 # max number of words in a question to use
        batch_size = 512 # how many samples to process at once
        n_epochs = 5 # how many times to iterate over all samples
        n_splits = 5 # Number of K-fold Splits
        SEED = 10
        debug = 0

        filter_sizes = [1,2,3,5]
        num_filters = 36
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, 1)


    def forward(self, x):
        x = self.embedding(x)  
        x = x.unsqueeze(1)  
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x)  
        return logit