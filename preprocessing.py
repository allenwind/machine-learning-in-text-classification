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

class Word2IdTransformer:
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