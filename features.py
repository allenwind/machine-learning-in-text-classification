import collections
import itertools
import numpy as np

__all__ = ["build_global_chars", "build_global_freq_chars",
           "VSMVectorizer", "CountVectorizer", "HashVectorizer",
           "CoveVectorizer", "TFIDFVectorizer", "NGramTransformer"]


def build_global_chars(X):
    # 建立全局词表
    return set(itertools.chain(*X))

def build_global_freq_chars(X):
    # 构建全局频率表
    return collections.Counter(itertools.chain(*X))

class VSMVectorizer:
    """向量空间模型"""

    def __init__(self, min_freq=0, max_freq=0, stopwords=set()):
        pass

    def fit(self, X, y=None):
        # 构建全局词表
        self.chars = set(itertools.chain(*X))
        # 给每个char一个全局索引
        self.char2id = {c:i for i, c in enumerate(self.chars)}

    def transform(self, X):
        vectors = np.zeros((len(X), len(self.chars)))
        for i, sentence in enumerate(X):
            for char in set(sentence):
                if char not in self.chars:
                    continue
                vectors[i, self.char2id[char]] = 1
        return vectors

    @property
    def vocab_size(self):
        return len(self.chars)

class CountVectorizer:
    """计算词全局频率"""

    def __init__(self, min_tf=16, max_tf=0, max_features=200, global_tf=True):
        self.min_tf = min_tf

    def fit(self, X, y=None):
        # 统计词频
        # 这种方法更简便
        # collections.Counter(itertools.chain(*X))
        freq_chars = collections.Counter(itertools.chain(*X))
        # 过滤低频词
        self.freq_chars = {c:f for c, f in freq_chars.items() if f >= self.min_tf}
        self.char2id = {c:i for i, c in enumerate(self.freq_chars)}

    def transform(self, X):
        vectors = np.zeros((len(X), len(self.freq_chars)))
        for i, sentence in enumerate(X):
            for char in set(sentence):
                if char not in self.freq_chars:
                    continue
                vectors[i, self.char2id[char]] = self.freq_chars[char]
        return vectors

    @property
    def vocab_size(self):
        return len(self.freq_chars)
    

class HashVectorizer:
    """Hashing Trick"""
    
    def __init__(self, min_freq=0):
        self.min_freq = min_freq

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass

    def _hashing(self, x):
        pass
        

class RandomVectorizer:
    """随机映射"""

    def __init__(self, min_freq=32, feature_size=128):
        self.min_freq = min_freq
        self.feature_size = feature_size

    def fit(self, X, y=None):
        freq_chars = collections.Counter(itertools.chain(*X))
        self.freq_chars = {c:f for c, f in freq_chars.items() if f >= self.min_freq}
        self.char2id = {c:i for i, c in enumerate(self.freq_chars)}
        self.randoms = np.random.normal(size=(len(self.char2id)+1, self.feature_size))

    def transform(self, X):
        vectors = np.zeros((len(X), self.feature_size))
        for i, sentence in enumerate(X):
            ids = [self.char2id.get(char, 0) for char in sentence]
            features = self.randoms[ids]
            vectors[i] = np.average(features, axis=0)
        return vectors

class GloveVectorizer:
    """共现特征:一个词的语义可以用它的上下文表示"""

    def fit(self, X, y=None):
        vocab = set(itertools.chain(*X))
        size = len(vocab)
        C = np.zeros((size, size))
        cf = collections.defaultdict(collections.Counter)
        for sentence in X:
            pass

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

class TFIDFVectorizer:
    """TF-IDF特征"""

    def __init__(self, min_freq=0):
        self.min_freq = min_freq

    def fit(self, X, y=None):
        # 计算全局idf
        self.vocab = set(itertools.chain(*X))
        self.id2char = {i:c for i, c in enumerate(self.vocab)}
        self.char2id = {c:i for i, c in self.id2char.items()}
        self.df = np.zeros((len(self.id2char,)))
        for i in range(len(self.df)):
            char = self.id2char[i]
            for sentence in X:
                if char in set(sentence):
                    self.df[i] += 1
        self.idf = np.log(len(X) / self.df)

    def transform(self, X):
        # 计算tf并与idf相乘
        vectors = np.zeros((len(X), len(self.vocab)))
        for i, sentence in enumerate(X):
            # char在sentence中的频率
            for char, freq in collections.Counter(sentence).items():
                if char not in self.vocab:
                    continue
                vectors[i, self.char2id[char]] = freq
        return vectors * self.idf

class NGramTransformer:
    """句子转换成ngram形式"""

    def __init__(self, ngram=2):
        self.ngram = ngram
    
    def transform(self, X):
        Xn = []
        for sentence in X:
            grams = []
            for i in range(len(sentence)-self.ngram+1):
                grams.append(sentence[i:i+n])
            Xn.append(grams)
        return Xn


if __name__ == "__main__":
    from dataset import load_weibo_senti_100k
    import matplotlib.pyplot as plt
    X, y = load_weibo_senti_100k()
    X = X[:2000]
    m = VSMVectorizer()
    m.fit(X)
    r = m.transform(X)
    plt.imshow(r)
    plt.show()

    m = CountVectorizer()
    m.fit(X)
    r = m.transform(X)
    plt.imshow(r)
    plt.show()

    m = RandomVectorizer()
    m.fit(X)
    r = m.transform(X)
    plt.imshow(r)
    plt.show()
