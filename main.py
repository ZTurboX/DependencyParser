import prepare_feature
import torch
import torch.nn as nn
import os
import sys
import argparse
import model
from const import Config
from torch import optim
import numpy as np
from Parsing import Parser,Decoding


config=Config()
parse=argparse.ArgumentParser()

parse.add_argument('--mode',default='test',help='train/valid/test')
parse.add_argument('--cuda',default=False)
args=parse.parse_args()

mode=args.mode
device=torch.device("cuda" if args.cuda else "cpu")

model=model.ParserModel(config)
if mode=="test":
    print("loading model...")
    state_dict=torch.load(open(config.model,'rb'))
    model.load_state_dict(state_dict)


def get_batch(input_data,y,batch_start,batch_size):

    '''
    word_input_batch=word_input[batch_start:batch_start+batch_size]
    pos_input_batch=pos_input[batch_start:batch_start+batch_size]
    label_input_batch=label_input[batch_start:batch_start+batch_size]
    y_batch=y[batch_start:batch_start+batch_size]

    word_input_batch=np.array(word_input_batch)
    pos_input_batch=np.array(pos_input_batch)
    label_input_batch=np.array(label_input_batch)


    word_input_batch=torch.LongTensor(word_input_batch).to(device)
    pos_input_batch=torch.LongTensor(pos_input_batch).to(device)
    label_input_batch=torch.LongTensor(label_input_batch).to(device)
    '''



    input_batch=input_data[batch_start:batch_start+batch_size]
    y_batch = y[batch_start:batch_start + batch_size]

    input_batch = np.array(input_batch)
    input_batch = torch.LongTensor(input_batch).to(device)


    y_batch=np.array(y_batch)
    y=np.zeros((y_batch.size,3))
    y[np.arange(y_batch.size),y_batch]=1

    return input_batch,y


def train_step(input_data,y,optimizer,criterion,dev_data,batch_size=config.batch_size):

    model.train()
    count=0
    total_loss=0
    for i in range(0, len(y), batch_size):
        optimizer.zero_grad()
        loss = 0
        '''
        word_input = input_data[0]
        pos_input = input_data[1]
        label_input = input_data[2]
        '''

        input_batch, y_batch = get_batch(input_data, y,i,batch_size)

        print("run minibatch :%d "%i)

        #input_batch,y_batch=get_batch(input_data,y,i,batch_size)
        y_batch=torch.from_numpy(y_batch.nonzero()[1]).long().to(device)
        y_predict_logits=model(input_batch)
        loss=criterion(y_predict_logits,y_batch)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
        print("minibatch : %d ,loss: %.5f" %(i,loss.item()))
        count+=1
    print("-----------------------------------------------")
    print("avg loss : %.5f"%(total_loss/count))
    print("-----------------------------------------------")

    print("Evauating on dev set...")

    model.eval()
    UAS=dev_step(dev_data)

    print("dev UAS :%.3f" % UAS)
    return UAS


def dev_step(dev_data,batch_size=config.batch_size):
    all_sentence=[]
    sentence2id={}
    for i,items in enumerate(dev_data):
        n_words=len(items["word"])-1
        sentence=[j+1 for j in range(n_words)]
        all_sentence.append(sentence)
        sentence2id[id(sentence)]=i

    decoding=Decoding(dev_data,sentence2id,model,device)
    dep=decoding.batch_parse(all_sentence,batch_size)

    print("calculate UAS......")
    UAS=all_items=0.0
    for i,items in enumerate(dev_data):
        head=[-1]*len(items["word"])
        for h,t in dep[i]:
            head[t]=h
        for pred_h,gold_h in zip(head[1:],items["head"][1:]):
            UAS+=1 if pred_h==gold_h else 0
            all_items+=1
    UAS/=all_items
    return UAS


def train(input_data,y,dev_data,batch_size=config.batch_size):
    best_UAS=0
    optimizer=optim.Adam(model.parameters())
    #optimizer = optim.Adagrad(model.parameters())
    criterion=torch.nn.CrossEntropyLoss()
    print("start train.....")
    for i in range(config.epoch_size):
        print("train epoch: %d"%i)
        UAS=train_step(input_data,y,optimizer,criterion,dev_data,batch_size)
        if UAS>best_UAS:
            best_UAS=UAS
            print("---------------------------------")
            print("best UAS :%.3f"%best_UAS)
            print("---------------------------------")
            print("saving model...")
            best_model_file=os.path.join(config.model_file,"epoch-%d_UAS-%.3f.pt"%(i,best_UAS))
            torch.save(model.state_dict(),best_model_file)


def test(test_data,batch_size=config.batch_size):
    model.eval()
    all_sentence = []
    sentence2id = {}
    for i, items in enumerate(test_data):
        n_words = len(items["word"]) - 1
        sentence = [j + 1 for j in range(n_words)]
        all_sentence.append(sentence)
        sentence2id[id(sentence)] = i

    decoding = Decoding(test_data, sentence2id, model, device)
    dep = decoding.batch_parse(all_sentence, batch_size)

    print("calculate UAS......")
    UAS = all_items = 0.0
    for i, items in enumerate(test_data):
        head = [-1] * len(items["word"])
        for h, t in dep[i]:
            head[t] = h
        for pred_h, gold_h in zip(head[1:], items["head"][1:]):
            UAS += 1 if pred_h == gold_h else 0
            all_items += 1
    UAS /= all_items
    print("test UAS : %.3f"%UAS)


if __name__=="__main__":

    feature=prepare_feature.Feature(config)
    if mode=='train':
        print("loading training data.....")
        train_data = feature.read_data(config.train_data_file)
        print("loading dev data.....")
        dev_data = feature.read_data(config.dev_data_file)
        input_data, t_input = feature.create_data(train_data)
        train(input_data,t_input,dev_data)
    elif mode=="test":
        print("loading test data.....")
        test_data = feature.read_data(config.test_data_file)
        test(test_data)




