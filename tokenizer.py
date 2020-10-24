import jieba

def jieba_tokenizer(s):
    # 分词
    return jieba.lcut(s)

def char_tokenizer(s):
    # 直接以字为单位
    return list(s)

def ngram_tokenizer(s, n=2):
    # n-gram 方式分词
    r = []
    if len(s) <= n:
        r.append(s)
        return r
    for i in range(len(s)-n+1):
        r.append(s[i:i+n])
    return r

import itertools
import numpy as np

class Transformer:

    def fix(self, X, y=None):
        # 参数学习
        pass

    def fit_transform(self, X, y=None):
        # 参数学习并返回变换结果
        pass

    def transform(self, X):
        # 返回变换结果
        pass

def build_vocab(X):
    vocab = set(itertools.chain(*X))
    vocab = {w:i for i, w in enumerate(vocab, start=1)}
    return vocab

class CharTokenizer:
    """字转id"""

    def __init__(self):
        self.word2id = {}
        self.UNKNOW = 0

    def fit(self, X):
        vocab = set(itertools.chain(*X))
        for i, w in enumerate(vocab, start=1):
            self.word2id[w] = i

    def transform(self, X):
        r = []
        for sample in X:
            s = []
            for w in sample:
                s.append(self.word2id.get(w, self.UNKNOW))
            r.append(s)
        return r

    def __len__(self):
        return len(self.word2id) + 1

    @property
    def vocab(self):
        return self.word2id
