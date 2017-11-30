import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import config
from dataLoader import Corpus
import os
import torch.optim
import pickle
import time


class RNNModel(nn.Module):
    def __init__(self, n_token):
        super(RNNModel, self).__init__()

        if config['use_dropout']:
            self.dropout = nn.Dropout(config['dropout_rate'])
        else:
            self.dropout = lambda x:x

        self.encoder = nn.Embedding(
            embedding_dim=config['embedding_size'],
            num_embeddings=n_token
        )

        self.encoder.weight.data.uniform_(
            -config['init_range'], config['init_range']
        )

        self.RNN = nn.LSTM(
            input_size=config['embedding_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['n_layers'],
            batch_first=True,
            bidirectional=config['bidirectional']
        )

        curr_dim = config['hidden_size'] * (config['bidirectional'] + 1)
        self.fc0 = nn.Linear(curr_dim, config['linear'][0])
        self.fc1 = nn.Linear(config['linear'][0], config['linear'][1])

    def forward(self, xs):
        embedding = self.dropout(self.encoder(xs))
        output, _ = self.RNN(embedding)
        x = self.dropout(output).view(-1, output.size(2))
        for idx in range(len(config['linear'])):
            x = F.relu(getattr(self, 'fc%d' % idx)(x))
        return x.view(-1, config['window_size'], 4)

