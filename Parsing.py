from prepare_feature import Feature
import model
import numpy as np
import torch
from const import Config

config=Config()
feature=Feature(config)


class Parser(object):
    def __init__(self,sentence):
        self.sentence=sentence
        self.stack=["<ROOT>"]
        self.buffer=list(sentence)
        self.dep=[]

    def parse_step(self,transition):
        if transition=="S" and len(self.buffer)>0:
            word=self.buffer.pop(0)
            self.stack.append(word)
        if transition=="L":
            head=self.stack[-1]
            dependent=self.stack.pop(-2)
            self.dep.append((head,dependent))
        if transition=='R':
            head=self.stack[-2]
            dependent=self.stack.pop()
            self.dep.append((head,dependent))

    def parse(self,transitions):
        for transition in transitions:
            self.parse_step(transition)
        return self.dep


class Decoding(object):
    def __init__(self,dataset,sentence2id,model,device):
        self.dataset=dataset
        self.sentence2id=sentence2id
        self.model=model
        self.device=device

    def predict(self,parsers):
        '''
         w_input=[]
        p_input=[]
        l_input=[]
        for p in parsers:
            word_features, pos_features, label_features=feature.create_features(p.stack, p.buffer, p.dep, self.dataset[self.sentence2id[id(p.sentence)]])
            w_input.append(word_features)
            p_input.append(pos_features)
            l_input.append(label_features)

        '''


        x=[feature.create_features(p.stack,p.buffer,p.dep,self.dataset[self.sentence2id[id(p.sentence)]]) for p in parsers]
        x=np.array(x).astype('int32')
        x=torch.from_numpy(x).long().to(self.device)

        '''
         w_input=np.array(w_input).astype('int32')
        p_input = np.array(p_input).astype('int32')
        l_input = np.array(l_input).astype('int32')

        w_input=torch.from_numpy(w_input).long().to(self.device)
        p_input=torch.from_numpy(p_input).long().to(self.device)
        l_input = torch.from_numpy(l_input).long().to(self.device)
        '''

        l=[feature.legal_labels(p.stack, p.buffer) for p in parsers]
        predict_logits=self.model(x)
        predict_logits=predict_logits.detach().numpy()
        #predict=np.argmax(predict_logits,1)
        predict = np.argmax(predict_logits+ 10000 * np.array(l).astype('float32'), 1)
        predict=["S" if p==2 else ("L" if p==0 else "R") for p in predict]
        return predict

    def batch_parse(self,sentences,batch_size):
        num_sentence=len(sentences)
        all_parsers=[Parser(sentence) for sentence in sentences]
        need_parse=all_parsers[:]
        dependent=[]
        while len(need_parse)>0:
            parsers=need_parse[:batch_size]
            predict_transitions=self.predict(parsers)
            for p,transition in zip(parsers,predict_transitions):
                p.parse([transition])
                if len(p.buffer)==0 and len(p.stack)==1:
                    need_parse.remove(p)

        dependent=[p.dep for p in all_parsers]
        return dependent











