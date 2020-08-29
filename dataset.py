import re
import pandas as pd

def load_weibo_senti_100k(file="dataset/weibo_senti_100k.csv", noe=True):
    df = pd.read_csv(file)
    X = df.review.to_list()
    y = df.label.to_list()
    # remove emoji tags
    if noe:
        X = [re.sub("\[.+?\]", lambda x:"", s) for s in X]
    return X, y

def load_stop_words(file="dataset/stop_words.txt"):
    with open(file, "r") as fp:
        stopwords = fp.read().splitlines()
    return set(stopwords)

