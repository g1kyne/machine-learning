#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC


"""将32x32的二进制图像转换为1x1024的向量"""
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)  # 打开文件
    for i in range(32):  # 按行读取
        # 读一行数据
        linestr = fr.readline()
        # 将每行的32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32*i+j] = int(linestr[j])
    return returnVect


"""手写数字分类测试"""
def handwritingClassTest():
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')
    # 文件夹下文件个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵，测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集类别
    for i in range(m):
        # 获得文件名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector('trainingDigits/%s'%(fileNameStr))
    # 设置分类器
    clf = SVC(C=200,kernel='rbf')
    # 训练分类器
    clf.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件名
        fileNameStr = testFileList[i]
        # 获得分类数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1x1024向量，用于测试
        vectorUnderTest = img2vector('testDigits/%s'%(fileNameStr))
        # 获得预测结果
        classfierResult = clf.predict(vectorUnderTest)
        print('分类返回结果为%d\t真实结果为%d'%(classfierResult, classNumber))
        if classfierResult != classNumber:
            errorCount += 1.0
    print('总共错了%d个数据\n错误率为%f%%'%(errorCount, errorCount/mTest*100))


if __name__ == '__main__':
    handwritingClassTest()
