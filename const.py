import os
import json


class Config:
    def __init__(self):
        home = './'
        data_home = os.path.join(home, 'data')
        self.vocab_file = os.path.join(data_home, 'vocab.json')
        self.train_data_file = os.path.join(data_home, 'train.json')
        self.dev_data_file = os.path.join(data_home, 'dev.json')
        self.test_data_file = os.path.join(data_home, 'test.json')
        self.vector_file = os.path.join(data_home, 'vectors-100.txt')

        self.embedding_size = 100
        self.batch_size = 2048
        self.dropout = 0.5
        self.hidden_size = 200
        self.epoch_size = 100

        self.vocab = json.load(open(self.vocab_file, 'r', encoding='utf-8'))
        self.words_number = len(self.vocab)

        self.model_file = os.path.join(data_home, 'output')

        self.model = os.path.join(data_home, 'model', 'model.pt')
