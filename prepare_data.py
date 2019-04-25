import json


def get_split_sentence(raw_data_file,split_data_file):
    sentence=[]
    with open(raw_data_file,'r',encoding='utf-8') as f:
        split_word=[]
        for line in f.readlines():
            item=line.strip().split()
            if len(item)==10:
                word=item[1]
                split_word.append(word)
            else:
                sentence.append(split_word)
                split_word=[]
    f.close()
    with open(split_data_file,'a',encoding='utf-8') as fs:
        for s in sentence:
            new_s=' '.join(s)
            fs.writelines(new_s)
            fs.writelines('\n')
    fs.close()

def get_data(raw_data_file,new_data_file):
    sentence=[]
    with open(raw_data_file,'r',encoding='utf-8') as f:
        word,pos,head,label=[],[],[],[]
        for line in f:
            item=line.strip().split()
            if len(item)==10:
                word.append(item[1])
                pos.append(item[4])
                head.append(int(item[6]))
                label.append(item[7])
            else:
                sentence.append({'word':word,'pos':pos,'head':head,'label':label })
                word, pos, head, label = [], [], [], []
    f.close()

    with open(new_data_file,'a',encoding='utf-8') as fs:
        for s in sentence:
            json.dump(s,fs,ensure_ascii=False)
            fs.writelines('\n')
    fs.close()
    print(len(sentence))

def get_vocab(data_file,vocab_file):
    words=["<UNK>","<ROOT>","<NULL>","<P><UNK>","<P><NULL>","<P><ROOT>","<l><NULL>"]
    words_num, pos_num, label_num = 0, 0, 0
    with open(data_file,'r',encoding='utf-8') as f:
        for line in f:
            data=json.loads(line)
            word=data["word"]
            pos=data["pos"]
            label=data["label"]

            for w in word:
                if w not in words:
                    words.append(w)
                    words_num+=1

            for p in pos:
                if "<p>"+p not in words:
                    words.append("<p>"+p)
                    pos_num+=1

            for l in label:
                if "<l>"+l not in words:
                    words.append("<l>"+l)
                    label_num+=1

    f.close()
    words2id={j:i for i,j in enumerate(words)}
    with open(vocab_file,'w',encoding='utf-8') as fs:
        fs.write(json.dumps(words2id,ensure_ascii=False,indent=4))
    fs.close()

    print("words size [%d]" % words_num)
    print("pos size [%d]" % pos_num)
    print("label size [%d]" % label_num)
    '''
    words size [34571]
    pos size [31]
    label size [11]
    '''



def data2id(data_file,data2id_file):
    with open("./raw_data/vocab.json",'r',encoding='utf-8') as f:
        vocab=json.load(f)
    f.close()
    data2id=[]
    with open(data_file,'r',encoding='utf-8') as fs:
        for line in fs:
            data=json.loads(line)
            word=[vocab["<ROOT>"]]+[vocab[w] if w in vocab else vocab["<UNK>"] for w in data["word"]]
            pos=[vocab["<P><ROOT>"]]+[vocab["<p>"+p] if "<p>"+p in vocab else vocab["<P><UNK>"] for p in data["pos"]]
            head=[-1]+data["head"]
            label=[-1]+[vocab["<l>"+l] if "<l>"+l in vocab else -1 for l in data["label"]]
            data2id.append({"word":word,"pos":pos,"head":head,"label":label})
    fs.close()

    with open(data2id_file,'a',encoding="utf-8") as fw:
        for item in data2id:
            json.dump(item,fw,ensure_ascii=False)
            fw.write("\n")
    fw.close()




if __name__=='__main__':
    raw_data_file='./data/train.conll'
    split_data_file='./data/split_sentence.txt'
    #get_split_sentence(raw_data_file,split_data_file)
    new_train_data_file='./raw_data/train.json'
    #get_data(raw_data_file,new_train_data_file)
    vocab_file='./raw_data/vocab.json'
    #get_vocab(new_train_data_file,vocab_file)
    train_data2id_file="./data/train.json"
    #data2id(new_train_data_file,train_data2id_file)
    raw_dev_file='./raw_data/dev.conll'
    new_dev_file='./raw_data/dev.json'
    #get_data(raw_dev_file, new_dev_file)
    raw_test_file='./raw_data/test.conll'
    new_test_file='./raw_data/test.json'
    #get_data(raw_test_file, new_test_file)
    data2id_dev_file='./data/dev.json'
    data2id_test_file='./data/test.json'
    #data2id(new_dev_file, data2id_dev_file)
    data2id(new_test_file, data2id_test_file)



