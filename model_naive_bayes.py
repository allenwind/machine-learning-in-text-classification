import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

from dataset import load_weibo_senti_100k, load_stop_words
from evaluation import plot_rocs, plot_pcs
from tokenizer import jieba_tokenizer, char_tokenizer

# 读取数据
X, y = load_weibo_senti_100k(noe=False)
stop_words = load_stop_words()

# 交叉验证
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# 特征提取 + 数据标准化
pipe = Pipeline([('count', CountVectorizer(max_features=100, 
                                           tokenizer=jieba_tokenizer, 
                                           stop_words=stop_words, 
                                           min_df=200)),
                 ('tf-idf', TfidfTransformer()),
                 ('norm', Normalizer()),
                 ])

X_train = pipe.fit_transform(X_train).toarray()
print("train size", X_train.shape)
X_test = pipe.transform(X_test).toarray()

# 训练
model = CategoricalNB()
model.fit(X_train, y_train)

# 模型评估
y_train_pred = model.predict_proba(X_train)[:, 1]
y_test_pred = model.predict_proba(X_test)[:, 1]
plot_rocs([y_train, y_test], [y_train_pred, y_test_pred], ["train", "test"])
plot_pcs([y_train, y_test], [y_train_pred, y_test_pred], ["train", "test"])
