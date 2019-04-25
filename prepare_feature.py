import json

import const



class Feature():
    def __init__(self,cfg:const.Config):
        self.transition=['L','R','S']
        self.tran2id={j:i for (i,j) in enumerate(self.transition)}
        self.id2tran={i:j for (i,j) in enumerate(self.transition)}
        self.vocab=cfg.vocab


    def read_data(self,data_file):
        data=[]
        with open(data_file,'r',encoding='utf-8') as f:
            for line in f:
                item=json.loads(line)
                data.append(item)
        f.close()
        return data

    def get_transition(self,stack,buffer,item):
        if len(stack)<2:
            return 2

        s1=stack[-1]
        s2=stack[-2]
        h1=item["head"][s1]
        h2=item["head"][s2]

        if s2>0 and h2==s1:
            return 0
        elif s1>0 and h1==s2 and (not any([x for x in buffer if item["head"][x]==s1])):
            return 1
        else:
            return None if len(buffer)==0 else 2

    def get_left_child(self,item,arcs):
        return sorted([arc[1] for arc in arcs if arcs[0]==item and arc[1]<item])

    def get_right_child(self,item,arcs):
        return sorted([arcs[1] for arc in arcs if arcs[0]==item and arc[1]>item],reverse=True)


    def create_features(self,stack,buffer,arcs,item):

        if stack[0]=='<ROOT>':
            stack[0]=0
        word_features=[]
        pos_features=[]
        label_features=[]
        features=[]

        word_features.extend([self.vocab["<NULL>"]]*(3-len(stack))+[item["word"][w] for w in stack[-3:]])
        word_features.extend([item["word"][w] for w in buffer[:3]]+[self.vocab["<NULL>"]]*(3-len(buffer)))

        pos_features.extend([self.vocab["<P><NULL>"]]*(3-len(stack))+[item["pos"][p] for p in stack[-3:]])
        pos_features.extend([item["pos"][p] for p in buffer[:3]]+[self.vocab["<P><NULL>"]]*(3-len(buffer)))

        for i in range(2):
            if i<len(stack):
                s=stack[-i-1]
                lc=self.get_left_child(s,arcs)
                rc=self.get_right_child(s,arcs)
                llc=self.get_left_child(lc[0]) if len(lc)>0 else []
                rrc=self.get_right_child(rc[0]) if len(rc)>0 else []

                word_features.append(item["word"][lc[0]] if len(lc)>0 else self.vocab["<NULL>"])
                word_features.append(item["word"][rc[0]] if len(rc) > 0 else self.vocab["<NULL>"])
                word_features.append(item["word"][lc[1]] if len(lc) > 1 else self.vocab["<NULL>"])
                word_features.append(item["word"][rc[1]] if len(rc) > 1 else self.vocab["<NULL>"])
                word_features.append(item["word"][llc[0]] if len(llc) > 0 else self.vocab["<NULL>"])
                word_features.append(item["word"][rrc[0]] if len(rrc) > 0 else self.vocab["<NULL>"])

                pos_features.append(item["pos"][lc[0]] if len(lc)>0 else self.vocab["<P><NULL>"])
                pos_features.append(item["pos"][rc[0]] if len(rc) > 0 else self.vocab["<P><NULL>"])
                pos_features.append(item["pos"][lc[1]] if len(lc) > 1 else self.vocab["<P><NULL>"])
                pos_features.append(item["pos"][rc[1]] if len(rc) > 1 else self.vocab["<P><NULL>"])
                pos_features.append(item["pos"][llc[0]] if len(llc) > 0 else self.vocab["<P><NULL>"])
                pos_features.append(item["pos"][rrc[0]] if len(rrc) > 0 else self.vocab["<P><NULL>"])

                label_features.append(item["label"][lc[0]] if len(lc)>0 else self.vocab["<l><NULL>"])
                label_features.append(item["label"][rc[0]] if len(rc) > 0 else self.vocab["<l><NULL>"])
                label_features.append(item["label"][lc[1]] if len(lc) > 1 else self.vocab["<l><NULL>"])
                label_features.append(item["label"][rc[1]] if len(rc) > 0 else self.vocab["<l><NULL>"])
                label_features.append(item["label"][llc[0]] if len(llc) > 0 else self.vocab["<l><NULL>"])
                label_features.append(item["label"][rrc[0]] if len(rrc) > 0 else self.vocab["<l><NULL>"])
            else:
                word_features.extend([self.vocab["<NULL>"]]*6)
                pos_features.extend([self.vocab["<P><NULL>"]]*6)
                label_features.extend([self.vocab["<l><NULL>"]]*6)

        features+=word_features
        features+=pos_features
        features+=label_features

        #return [word_features,pos_features,label_features],features
        return features


    def create_data(self,data):

        words_input, pos_input, label_input, t_input = [], [], [], []
        input=[]
        for id,item in enumerate(data):
            n_words=len(item["word"])-1
            stack=[0]
            buffer=[i+1 for i in range(n_words)]
            arcs=[]
            for i in range(n_words*2):
                gold_t=self.get_transition(stack,buffer,item)
                if gold_t is None:
                    break
                #word_features,pos_features,label_features=self.create_features(stack,buffer,arcs,item)
                features = self.create_features(stack, buffer, arcs, item)
                if gold_t==2:
                    stack.append(buffer[0])
                    buffer=buffer[1:]
                elif gold_t==0:
                    arcs.append((stack[-1],stack[-2],gold_t))
                    stack=stack[:-2]+[stack[-1]]
                elif gold_t==1:
                    arcs.append((stack[-2],stack[-1],gold_t))
                    stack=stack[:-1]
                '''
                words_input.append(word_features)
                pos_input.append(pos_features)
                label_input.append(label_features)
                t_input.append(gold_t)
                '''
                input.append(features)
                t_input.append(gold_t)

        #return [words_input,pos_input,label_input],t_input
        return input,t_input







