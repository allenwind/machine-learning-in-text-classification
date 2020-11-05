import numpy as np

# 纯 numpy 实现的 Logistic Regression

class LogisticRegression:

    def __init__(self):
        self.lr = 0.001
        self.batch_size = 1
        self.epochs = 5000

    def fit(self, X, y, batch_size=None, epochs=None):
        # 初始化权重
        self.weights = np.zeros(len(X[0])+1)
        X = np.hstack([X, np.ones((len(X), 1))])

        p_count = 0
        t = 0
        while t <= self.epochs:
            index = np.random.randint(0, len(y) - 1)
            sample = X[index]
            label = y[index]

            if label == self._predict_one(sample):
                p_count += 1
                if p_count > self.epochs:
                    break
                continue

            t += 1
            p_count = 0

            wx = np.dot(self.weights, sample)
            exp_wx = np.exp(wx)
            # 梯度下降更新参数
            dl = -label * sample + sample * exp_wx / (1 + exp_wx)
            self.weights -= self.lr * dl

    def _predict_one(self, sample):
        wx = np.dot(self.weights, sample)
        exp_wx = np.exp(wx)
        p1 = exp_wx / (1 + exp_wx)
        p2 = 1 - p1
        if p1 > p2:
            return 1
        return 0

    def predict(self, X):
        X = np.hstack([X, np.ones((len(X), 1))])
        y = []
        for sample in X:
            y.append(self._predict_one(sample))
        return np.array(y)

    def _predict_one_proba(self, sample):
        wx = np.dot(self.weights, sample)
        exp_wx = np.exp(wx)
        p = exp_wx / (1 + exp_wx)
        return p

    def predict_proba(self, X):
        X = np.hstack([X, np.ones((len(X), 1))])
        y = []
        for sample in X:
            y.append(self._predict_one_proba(sample))
        return np.array(y)


def test():
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    lr = LogisticRegression()
    lr.fit(X, y)
    y_pred = lr.predict_proba(X)

    import matplotlib.pyplot as plt
    plt.plot(y_pred)
    plt.show()

if __name__ == "__main__":
    test()

