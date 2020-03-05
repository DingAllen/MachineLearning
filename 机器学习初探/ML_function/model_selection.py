import numpy as np

def train_test_split(X, Y, test_ratio = 0.2, seed = None):
    """将数据X和Y按照test_ratio分割成X_train，X_test，Y_train，Y_test"""

    assert X.shape[0] == Y.shape[0], "the size of X must be equal to the size of Y"
    assert 0.0 <= test_ratio <= 1.0, "test_ration must be vaild"

    if seed:
        np.random.seed(seed)

    shuffle_indexs = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexs = shuffle_indexs[:test_size]
    train_indexs = shuffle_indexs[test_size:]

    X_train = X[train_indexs]
    Y_train = Y[train_indexs]

    X_test = X[test_indexs]
    Y_test = Y[test_indexs]

    return X_train, X_test, Y_train, Y_test


