import torch
import torch.nn as nn
import const
import numpy as np
import json
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class Cube(nn.Module):
    def __init__(self,w_features,p_features,l_features,out_features):
        super(Cube, self).__init__()
        self.w_features=w_features
        self.p_features=p_features
        self.l_features=l_features
        self.w_weight=Parameter(torch.Tensor(out_features,w_features))
        self.p_weight=Parameter(torch.Tensor(out_features,p_features))
        self.l_weight=Parameter(torch.Tensor(out_features,l_features))
        self.bias=Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv=1./math.sqrt(self.w_weight.size(1))
        p_stdv=1./math.sqrt(self.p_weight.size(1))
        l_stdv=1./math.sqrt(self.l_weight.size(1))

        self.w_weight.data.uniform_(-w_stdv,w_stdv)
        self.p_weight.data.uniform_(-p_stdv,p_stdv)
        self.l_weight.data.uniform_(-l_stdv,l_stdv)
        self.bias.data.uniform_(-w_stdv,w_stdv)

    def forward(self,w_input,p_input,l_input):
        #o1=torch.add(torch.matmul(w_input,self.w_weight),torch.matmul(p_input,self.p_weight))
        #o2=torch.add(o1,torch.matmul(l_input,self.l_weight))
        o1=F.linear(w_input,self.w_weight)
        o2=F.linear(p_input,self.p_weight)
        o3=F.linear(l_input,self.l_weight)
        o=torch.pow(o1+o2+o3+self.bias,3)

        #res=torch.pow(o2+self.bias,3)
        return o


class ParserModel(nn.Module):
    def __init__(self,config:const.Config):
        super(ParserModel, self).__init__()
        self.hidden_size=config.hidden_size
        self.embedding_size=config.embedding_size
        self.words_number=config.words_number
        self.vocab=config.vocab
        self.dropout=nn.Dropout(config.dropout)
        self.word_embedding=nn.Embedding(self.words_number,self.embedding_size)
        self.load_pretrained_embedding(config)
        #self.word_embedding.weight.requires_grad=True

        self.hidden_layer=nn.Linear(self.embedding_size*48,self.hidden_size)
        #self.cube = Cube(self.embedding_size * 18, self.embedding_size * 18, self.embedding_size * 12, self.hidden_size)
        self.out=nn.Linear(self.hidden_size,3)



    def load_pretrained_embedding(self,config:const.Config):
        words_vectors={}
        for line in open(config.vector_file,encoding='utf-8').readlines():
            items=line.strip().split()
            words_vectors[items[0]]=[float(x) for x in items[1:]]
        embeddding_matrix=np.asarray(np.random.normal(0,0.9,(self.words_number,100)),dtype='float32')

        for word in self.vocab:
            if word in words_vectors:
                embeddding_matrix[self.vocab[word]]=words_vectors[word]
        self.word_embedding.weight=nn.Parameter(torch.tensor(embeddding_matrix))


    def forward(self,input_data):
        x=self.word_embedding(input_data)
        x=x.view(x.size()[0],-1)
        x = self.dropout(x)
        x=self.hidden_layer(x)
        #hidden=torch.pow(x,3)
        hidden=F.relu(x)


        '''
        w=self.word_embedding(w_input)
        p=self.word_embedding(p_input)
        l=self.word_embedding(l_input)

        w=w.view(w.size()[0],-1)
        p=p.view(p.size()[0],-1)
        l=l.view(l.size()[0],-1)

        hidden=self.cube(w,p,l)
        hidden=self.dropout(hidden)
        '''

        y_predict_logits = self.out(hidden)
        return y_predict_logits

