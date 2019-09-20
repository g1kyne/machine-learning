# -*- coding: UTF-8 -*-
import re
import numpy as np
import random


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 与词汇表等长的向量，存放1 0 ，是否出现
    for word in inputSet:  # 遍历输入向量的所有单词，若出现，将returnVec置1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("单词：%s 不在词汇表中！" % word)
    return returnVec


"""词袋模型：每个单词可以出现多次
    与setOfWords2Vec不同的是 每当遇到一个单词，会增加词向量对应值，而不是设为1"""
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 与词汇表等长的向量，存放1 0 ，是否出现
    for word in inputSet:  # 遍历输入向量的所有单词，若出现，将returnVec置1
        if word in vocabList:
            # returnVec[vocabList.index(word)] = 1
            returnVec[vocabList.index(word)] += 1
        else:
            print("单词：%s 不在词汇表中！" % word)
    return returnVec


"""将切分的实验样本词条整理成不重复的词汇表
    这一步是为了产生一个大而全的集合，这个集合包括了所有文档（即第一步产生的6个文档）中的词条，但每个词条都不重复"""
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的集合，存储dataSet中出现的词
    for ducument in dataSet:  # 遍历数据集中的每一个文本
        vocabSet = vocabSet | set(ducument)  # 与集合求并
    return list(vocabSet)  # 返回词汇表


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
    # print('p0: ', p0)
    # print('p1: ', p1)
    if p1 > p0:
        return 1
    else:
        return 0


"""接收一个大字符串，并解析为字符串列表"""
def textParse(bigDtring):
    listOfTokens = re.split(r'\W', bigDtring)  # 以非字母数字下划线切分大字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


"""垃圾邮件分类测试器"""
def spamTest():
    docList = []; classList = []

    # 处理25个垃圾邮件和非垃圾邮件，生成分割后的词汇列表和分类列表
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)  # 将每个txt处理后的词汇列表加到docList中  doclist = [ [],[],[],...]
        classList.append(1)  # spam中为垃圾邮件，分类列表设为1
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        classList.append(0)

    # 建立词汇表
    vocabList = createVocabList(docList)

    # 没分出测试集前的训练集有50个样本，测试集0个样本,trainingSet中保存的是50个样本的索引值
    trainingSet = list(range(50)); testSet = []

    # 从50个训练集中随机分出10个作为测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))  # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
        testSet.append(randIndex)
        del(trainingSet[randIndex])

    # 将训练集转化为用1和0表示的词向量[1,0,0,...],[1,0,0,....],...   trainClasses为该40个训练样本对应的分类
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    # 得到了训练样本的词向量和对应类别，就可以训练贝叶斯模型
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

    # 给测试集分类并计算错误率
    errorCount = 0
    for testIndex in testSet:
        wordVec = setOfWords2Vec(vocabList, docList[testIndex])  # 将testSet中的下标对应的词汇列表 转化为1和0表示词向量
        classifyResult = classifyNB(np.array(wordVec), p0V, p1V, pSpam)  # 使用贝叶斯模型分类的结果
        if classifyResult != classList[testIndex]:  # 如果与实际分类不一样，错误数+1
            errorCount += 1
            print('分类错误的数据集为：', docList[testIndex])
    print('错误率为：', errorCount / len(testSet))


if __name__ == '__main__':
    spamTest()



