#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# https://blog.csdn.net/c406495762/article/details/77723333
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.font_manager import FontProperties


"""Sigmoid函数"""
def sigmoid(intX):
    return 1.0 / (1+np.exp(-intX))


"""梯度上升算法找到最佳回归系数  P78页"""
def gradAscent(dataList, labelList):
    dataMat = np.mat(dataList)
    labelMat = np.mat(labelList).transpose()
    m, n = dataMat.shape
    weights = np.ones((n, 1))  # 回归系数矩阵初始化为1
    weights_arr = np.array([])  # 用于画图
    alpha = 0.001
    R = 500  # 重复次数
    for i in range(R):
        h = sigmoid(dataMat * weights)
        error = labelMat - h
        weights = weights + alpha * dataMat.transpose() * error
        weights_arr = np.append(weights_arr, weights)
    weights_arr = weights_arr.reshape(R, n)
    return weights.getA(), weights_arr


"""改进的随机梯度上升算法：新增迭代次数，随机选择一个样本"""
def stocGradAscent1(dataMat, labelList, numIter=150):  # 新增迭代次数
    m, n = dataMat.shape
    weights = np.ones(n)  # [1,1,1]
    for j in range(numIter):  # 迭代numIter次
        dataIndex = list(range(m))  # 生成0到99的列表
        for i in range(m):  # 遍历每个样本
            alpha = 4/(1.0 + j + i) + 0.01  # 降低alpha大小
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机生成一个0到未被选择列表长度范围的数作为索引
            h = sigmoid(sum(dataMat[randIndex] * weights))  # 计算该样本的h
            error = labelList[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]  # 迭代出weights
            del(dataIndex[randIndex])  # 删除已经被选到的值
    return weights


"""分类函数"""
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


"""使用Python写的Logistic分类器做预测"""
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    # 处理训练集数据
    TrainingSet = []
    TrainingLabels = []
    for line in frTrain.readlines():
        line = line.strip().split('\t')
        TrainingLine = []
        for i in range(len(line) - 1):
            TrainingLine.append(float(line[i]))
        TrainingSet.append(TrainingLine)
        TrainingLabels.append(float(line[-1]))
    # 得到回归系数
    weights = stocGradAscent1(np.array(TrainingSet), TrainingLabels, 500)
    # 处理测试集数据
    TestSet = []
    TestLabels = []
    error_count = 0
    test_count = 0
    for line in frTest.readlines():
        line = line.strip().split('\t')
        test_count += 1
        TestLine = []
        for i in range(len(line) - 1):
            TestLine.append(float(line[i]))
        if int(classifyVector(TestLine, weights)) != int(line[-1]):
            error_count += 1
    print('错误率为：', float(error_count/test_count) * 100)
    # classifier = LogisticRegression(solver='liblinear',max_iter=10).fit(trainingSet, trainingLabels)
    # test_accurcy = classifier.score(testSet, testLabels) * 100
    # print('正确率:%f%%' % test_accurcy)


if __name__ == '__main__':
    colicTest()















