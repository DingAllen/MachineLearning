import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score


class kNNClassifier:

    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be vaild"
        self.k = k
        self._X_train = None
        self._Y_train = None

    def fit(self, X_train, Y_train):
        """根据训练数据集X_train和Y_train训练kNN分类器"""

        assert X_train.shape[0] == Y_train.shape[0], "the size of X_train must equal to the size of Y_train"

        self._X_train = X_train
        self._Y_train = Y_train

        return self

    def predict(self, X_predict):
        """给定待预测X_predict，返回表示X_predict的结果向量"""

        assert self._X_train is not None and self._Y_train is not None, "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[
            1], "the feature number of X_predict of X_predict must be equal to X_train"

        Y_predict = [self._predict(x) for x in X_predict]

        return np.array(Y_predict)

    def _predict(self, x):
        """给定单个待预测数据x，返回x的预测结果值"""
        assert self._X_train.shape[1] == x.shape[0], "the feature number of x must equal to X_train"

        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._Y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "kNN(k = %d)" % self.k
