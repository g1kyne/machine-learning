# -*- coding: UTF-8 -*-
# https://blog.csdn.net/c406495762/article/details/77341116
import numpy as np
import math


"""创建实验样本
    其中的listOPosts即list Of Posts，文档列表，就是帖子列表、邮件列表等等。你可以认为列表中的一元素就是一个帖子或者回复，
    在此例中一共6个文档、帖子、回复（以后统称文档）
    可以看到，2、4、6句stupid，garbage存在侮辱性词条，第1、3、5个句子，不存在侮辱性词条，所以，对应的类别标签设置为
    listClasses = [0,1,0,1,0,1]"""
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],   # 切分的词条
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]   # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec


"""将切分的实验样本词条整理成不重复的词汇表
    这一步是为了产生一个大而全的集合，这个集合包括了所有文档（即第一步产生的6个文档）中的词条，但每个词条都不重复"""
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的集合，存储dataSet中出现的词
    for ducument in dataSet:  # 遍历数据集中的每一个文本
        vocabSet = vocabSet | set(ducument)  # 与集合求并
    return list(vocabSet)  # 返回词汇表


"""根据vocabList词汇表，将inputSet向量化，向量的每个元素为1(出现在词汇表)或0(没出现)
    该函数的输入参数为词汇表及某个文档，输出的是文档向量，
    向量的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 与词汇表等长的向量，存放1 0 ，是否出现
    for word in inputSet:  # 遍历输入向量的所有单词，若出现，将returnVec置1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("单词：%s 不在词汇表中！" % word)
    return returnVec


"""trainMatrix就是由各个文档转化成的词向量构成的矩阵，而trainCategory就是这几个文档的类别，也就是这几个文档是不是含有侮辱性词条"""
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 训练文档数目，此处为 6
    numWords = len(trainMatrix[0])  # 每篇文档的个数 32个1和0 表示出现或未出现
    pAbusive = sum(trainCategory)/len(trainCategory)  # 属于侮辱类(Abusive)文档的概率 即P(c1)
    p0Num = np.ones(numWords)  # 创建0数组
    p1Num = np.ones(numWords)
    p0Denom = 2.0  # Denom分母  这里防止了概率为0的情况
    p1Denom = 2.0
    for i in range(numTrainDocs):  # 遍历每个文档
        if trainCategory[i] == 1:  # 若该文档为侮辱性文档
            p1Num += trainMatrix[i]  # 词向量叠加，可计算某词在侮辱类中出现了多少次
            p1Denom += sum(trainMatrix[i])  # 共出现了多少个单词
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)  # 类别为1（即侮辱类）的条件概率（没个词的出现次数/总的出现个数）
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


"""使用贝叶斯分类"""
def classifyNB(classifyVec, p0Vec, p1Vec, pAbusive):
    p1 = sum(classifyVec * p1Vec) + np.log(pAbusive)  # 这里为对应元素相乘
    p0 = sum(classifyVec * p0Vec) + np.log(1.0 - pAbusive)
    print('p0: ', p0)
    print('p1: ', p1)
    if p1 > p0:
        return 1
    else:
        return 0


"""测试贝叶斯分类器"""
def testingNB():
    listOPosts, listClasses = loadDataSet()									#创建实验样本
    myVocabList = createVocabList(listOPosts)								#创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))				#将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))		#训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']									#测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))				#测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')										#执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')										#执行分类并打印分类结果

    testEntry = ['stupid', 'garbage']										#测试样本2
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))				#测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')										#执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')										#执行分类并打印分类结果

# 测试
if __name__ == '__main__':
    # postingList, classVec = loadDataSet()
    # for posting in postingList:
    #     print(posting)
    # print(classVec)
    # 输出
    # ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
    # ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid']
    # ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him']
    # ['stop', 'posting', 'stupid', 'worthless', 'garbage']
    # ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him']
    # ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    # [0, 1, 0, 1, 0, 1]  存放每个词条所属类别，1为侮辱性，0为正常


    # print("词条向量表:\n", postingList)  # print()中 +必须是同类型数据，  ，可以是不同类型
    # [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], ['
    # myVocabList = createVocabList(postingList)
    # print("词汇表：\n", myVocabList)
    # ['has', 'help', 'quit', 'dog', 'buying', 'not', 'so', 'posting', 'mr', 'stupid', 'is', 'cute', 'dalmation', 'please', 'steak', 'to', 'flea', 'park', 'maybe'
    # trainMat = []
    # for posting in postingList:  # 遍历词条向量表中每条向量，依次转化
    #     trainMat.append(setOfWords2Vec(myVocabList, posting))
    # print('训练集（01是否出现）:\n', trainMat)
    # [[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0,
    """ 从运行结果可以看出，postingList是原始的词条列表，myVocabList是词汇表。myVocabList是所有单词出现的集合，没有重复的元素。
    词汇表是用来干什么的？没错，它是用来将词条向量化的，一个单词在词汇表中出现过一次，那么就在相应位置记作1，如果没有出现就在相应位置记作0。
    trainMat是所有的词条向量组成的列表。它里面存放的是根据myVocabList向量化的词条向量
    原文链接：https://blog.csdn.net/c406495762/article/details/77341116"""
    testingNB()













