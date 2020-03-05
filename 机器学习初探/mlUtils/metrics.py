import numpy as np

def accuracy_score(y_true, y_predict):
    '''计算预测准确率'''
    assert y_true.shape[0] == y_predict.shape[0], "预测值必须与正确值尺度一致"

    return sum(y_true == y_predict) / len(y_true)

