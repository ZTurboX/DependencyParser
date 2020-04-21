import torch
import torch.nn as nn
import const
import numpy as np
import json
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class ParserModel(nn.Module):
    def __init__(self, config: const.Config):
        super(ParserModel, self).__init__()
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.words_number = config.words_number
        self.vocab = config.vocab
        self.dropout = nn.Dropout(config.dropout)
        self.word_embedding = nn.Embedding(self.words_number,
                                           self.embedding_size)
        self.load_pretrained_embedding(config)
        #self.word_embedding.weight.requires_grad=True

        self.hidden_layer = nn.Linear(self.embedding_size * 48,
                                      self.hidden_size)
       
        self.out = nn.Linear(self.hidden_size, 3)

    def load_pretrained_embedding(self, config: const.Config):
        words_vectors = {}
        for line in open(config.vector_file, encoding='utf-8').readlines():
            items = line.strip().split()
            words_vectors[items[0]] = [float(x) for x in items[1:]]
        embeddding_matrix = np.asarray(np.random.normal(
            0, 0.9, (self.words_number, 100)),
                                       dtype='float32')

        for word in self.vocab:
            if word in words_vectors:
                embeddding_matrix[self.vocab[word]] = words_vectors[word]
        self.word_embedding.weight = nn.Parameter(
            torch.tensor(embeddding_matrix))

    def forward(self, input_data):
        x = self.word_embedding(input_data)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        x = self.hidden_layer(x)

        hidden = F.relu(x)

        y_predict_logits = self.out(hidden)
        return y_predict_logits
