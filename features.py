import numpy as np
import collections
import itertools

class Transformer:

    def fix(self, X):
        # 参数学习
        pass

    def fit_transform(self, X):
        # 参数学习并返回变换结果
        pass

    def transform(self, X):
        # 返回变换结果
        pass

def global_tf(X):
    # 快速统计所有文档词频
    return collections.Counter(itertools.chain(*X))

def tf(X):
    vocab = set(itertools.chain(*X))
    # 给每个词一个全局索引
    w2i = {w:i for i, w in enumerate(vocab)}
    # 文档-单词矩阵
    tf = np.zeros((len(X), len(vocab)))
    for i, d in enumerate(X):
        c = collections.Counter(d)
        for w in c.keys():
            tf[i, w2i[w]] = c[w]
    return tf

def idf(X):
    vocab = set(itertools.chain(*X))
    i2w = {i:w for i, w in enumerate(vocab)}
    df = np.zeros((len(i2w),))
    for i in range(len(i2w)):
        for sample in X:
            for w in sample:
                if i2w[i] == w:
                    # 只需要记录一次
                    df[i] += 1
                    break
    idf = len(X) / np.log(df+1)
    return idf

def tf_idf(X):
    # 构造词表
    vocab = set(itertools.chain(*X))
    # 给每个词一个全局索引
    w2i = {w:i for i, w in enumerate(vocab)}
    tf = np.zeros((len(X), len(vocab)))
    for i, d in enumerate(X):
        c = collections.Counter(d)
        for w in c.keys():
            tf[i, w2i[w]] = c[w]

    i2w = {i:w for i, w in enumerate(vocab)}
    df = np.zeros((len(vocab),))
    for i in range(len(vocab)):
        for sample in X:
            for w in sample:
                if i2w[i] == w:
                    # 只需要记录一次
                    df[i] += 1
                    break
    idf = len(X) / np.log(df+1)
    return tf * idf

def cbow(X):
    pass

def ngram(X, n=1):
    pass

def vsm(X, max_freq=0, min_freq=0):
    # 全局词表
    vocab = set(itertools.chain(*X))
    # 给每个词一个全局索引
    w2i = {w:i for i, w in enumerate(vocab)}
    vs = np.zeros((len(X), len(vocab)))
    for i, sample in enumerate(X):
        for w in sample:
            vs[i, w2i[w]] = 1
    return vs

def test():
    from dataset import read_dataset
    import matplotlib.pyplot as plt
    data_path = "dataset/weibo_senti_100k.csv"
    X, y = read_dataset(data_path)
    X = [list(i) for i in X[:10000]]
    r = tf_idf(X)
    print(r.shape)
    plt.imshow(r)
    plt.show()

test()
