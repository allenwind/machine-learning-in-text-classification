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
