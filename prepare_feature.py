import json

import const


class Feature():
    def __init__(self, cfg: const.Config):
        self.transition = ['L', 'R', 'S']
        self.tran2id = {j: i for (i, j) in enumerate(self.transition)}
        self.id2tran = {i: j for (i, j) in enumerate(self.transition)}
        self.vocab = cfg.vocab

    def read_data(self, data_file):
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        f.close()
        return data

    def get_transition(self, stack, buffer, item):
        '''
        获取转移动作：
        left-arc:return 0
        right-arc:return 1
        shift:return 2
        '''
        if len(stack) < 2:
            return 2

        s1 = stack[-1]
        s2 = stack[-2]
        h1 = item["head"][s1]
        h2 = item["head"][s2]

        if s2 > 0 and h2 == s1:
            return 0
        elif s2 >= 0 and h1 == s2 and (not any(
            [x for x in buffer if item["head"][x] == s1])):
            return 1
        else:
            return None if len(buffer) == 0 else 2

    def get_left_child(self, item, arcs):
        return sorted(
            [arc[1] for arc in arcs if arc[0] == item and arc[1] < item])

    def get_right_child(self, item, arcs):
        return sorted(
            [arc[1] for arc in arcs if arc[0] == item and arc[1] > item],
            reverse=True)

    def create_features(self, stack, buffer, arcs, item):
        '''
        特征模板：
        (1)stack和buffer的前3个单词:s1、s2、s3、b1、b2、b3;
        (2)栈顶两个单词的第一个和第二个最左边/最右边的子单词:lc1(si)， rc1(si)， lc2(si)， rc2(si)， i = 1,2。
        (3)最左边的最左边/最右边的最右边——堆栈上最上面两个单词的大多数子元素:lc1(lc1(si))、rc1(rc1(si))、i = 1,2。
        '''

        if stack[0] == '<ROOT>':
            stack[0] = 0
        word_features = []
        pos_features = []
        label_features = []
        features = []

        word_features.extend([self.vocab["<NULL>"]] * (3 - len(stack)) +
                             [item["word"][w] for w in stack[-3:]])
        word_features.extend([item["word"][w] for w in buffer[:3]] +
                             [self.vocab["<NULL>"]] * (3 - len(buffer)))

        pos_features.extend([self.vocab["<P><NULL>"]] * (3 - len(stack)) +
                            [item["pos"][p] for p in stack[-3:]])
        pos_features.extend([item["pos"][p] for p in buffer[:3]] +
                            [self.vocab["<P><NULL>"]] * (3 - len(buffer)))

        for i in range(2):
            if i < len(stack):
                s = stack[-i - 1]
                lc = self.get_left_child(s, arcs)
                rc = self.get_right_child(s, arcs)
                llc = self.get_left_child(lc[0], arcs) if len(lc) > 0 else []
                rrc = self.get_right_child(rc[0], arcs) if len(rc) > 0 else []

                word_features.append(item["word"][lc[0]]
                                     if len(lc) > 0 else self.vocab["<NULL>"])
                word_features.append(item["word"][rc[0]]
                                     if len(rc) > 0 else self.vocab["<NULL>"])
                word_features.append(item["word"][lc[1]]
                                     if len(lc) > 1 else self.vocab["<NULL>"])
                word_features.append(item["word"][rc[1]]
                                     if len(rc) > 1 else self.vocab["<NULL>"])
                word_features.append(item["word"][llc[0]]
                                     if len(llc) > 0 else self.vocab["<NULL>"])
                word_features.append(item["word"][rrc[0]]
                                     if len(rrc) > 0 else self.vocab["<NULL>"])

                pos_features.append(item["pos"][lc[0]] if len(lc) > 0 else self
                                    .vocab["<P><NULL>"])
                pos_features.append(item["pos"][rc[0]] if len(rc) > 0 else self
                                    .vocab["<P><NULL>"])
                pos_features.append(item["pos"][lc[1]] if len(lc) > 1 else self
                                    .vocab["<P><NULL>"])
                pos_features.append(item["pos"][rc[1]] if len(rc) > 1 else self
                                    .vocab["<P><NULL>"])
                pos_features.append(item["pos"][llc[0]] if len(llc) > 0 else
                                    self.vocab["<P><NULL>"])
                pos_features.append(item["pos"][rrc[0]] if len(rrc) > 0 else
                                    self.vocab["<P><NULL>"])

                label_features.append(item["label"][lc[0]] if len(lc) > 0 else
                                      self.vocab["<l><NULL>"])
                label_features.append(item["label"][rc[0]] if len(rc) > 0 else
                                      self.vocab["<l><NULL>"])
                label_features.append(item["label"][lc[1]] if len(lc) > 1 else
                                      self.vocab["<l><NULL>"])
                label_features.append(item["label"][rc[1]] if len(rc) > 1 else
                                      self.vocab["<l><NULL>"])
                label_features.append(item["label"][llc[0]] if len(llc) > 0
                                      else self.vocab["<l><NULL>"])
                label_features.append(item["label"][rrc[0]] if len(rrc) > 0
                                      else self.vocab["<l><NULL>"])
            else:
                word_features.extend([self.vocab["<NULL>"]] * 6)
                pos_features.extend([self.vocab["<P><NULL>"]] * 6)
                label_features.extend([self.vocab["<l><NULL>"]] * 6)

        features += word_features
        features += pos_features
        features += label_features

        #return word_features,pos_features,label_features
        return features

    def legal_labels(self, stack, buffer):
        labels = [1] if len(stack) > 2 else [0]
        labels += [1] if len(stack) >= 2 else [0]
        labels += [1] if len(buffer) > 0 else [0]
        return labels

    def create_data(self, data):

        words_input, pos_input, label_input, t_input = [], [], [], []
        input = []
        for id, item in enumerate(data):
            n_words = len(item["word"]) - 1
            stack = [0]
            buffer = [i + 1 for i in range(n_words)]
            arcs = []
            for i in range(n_words * 2):
                gold_t = self.get_transition(stack, buffer, item)
                if gold_t is None:
                    break
                features = self.create_features(stack, buffer, arcs, item)
                if gold_t == 2:
                    # SHIFT:moves b1 from the buffer to the stack
                    stack.append(buffer[0])
                    buffer = buffer[1:]
                elif gold_t == 0:
                    # LEFT-ARC(l): adds an arc s1 → s2 with label l and removes s2 from the stack
                    arcs.append((stack[-1], stack[-2], gold_t))
                    stack = stack[:-2] + [stack[-1]]
                elif gold_t == 1:
                    # RIGHT-ARC(l): adds an arc s2 → s1 with  label l and removes s1 from the stack.
                    arcs.append((stack[-2], stack[-1], gold_t))
                    stack = stack[:-1]


                input.append(features)
                t_input.append(gold_t)

        return input, t_input
