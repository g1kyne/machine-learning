#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# https://blog.csdn.net/c406495762/article/details/77500679
import os
import jieba
import random
import operator
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB


"""数据集分类文件如下
    Sample下文件夹 C000008 财经
                   C000010 IT
                   C000013 健康
"""


"""文本预处理  把每个txt文档切分，并做好类别标记"""
def TextPProcessing0(folder_path):
    folder_list = os.listdir(folder_path)  # 查看folder_path文件夹
    data_list = []  # 训练集
    class_list = []  # 分类集

    # 遍历子文件夹,处理每个子文件夹
    for folder in folder_list:
        # 形成新路径：folder_path/folder
        # os.path.join(path1[,path2[,path3[,...[,pathN]]]])  将多个路径组合后返回
        new_folder_path = os.path.join(folder_path, folder)

        # 打开子文件夹中的TXT文件
        files = os.listdir(new_folder_path)

        # 处理子文件夹中的每个TXT文件
        for file in files:
            # 打开并读出txt文件内容
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:
                raw = f.read()

                # 文本切分
                word_cut = jieba.cut(raw, cut_all=False)  # 用精简模式切分，返回可迭代的generator
                word_list = list(word_cut)

                # 保存到数据集中
                data_list.append(word_list)
                class_list.append(folder)
            print(data_list)
            print(class_list)


"""升级版数据预处理：把所有文本分为训练集和测试集
    并对训练集所有单词按出现频率降序排列"""
def TextProcessing(folder_path, test_size = 0.2):  # 文件路径，测试集占比默认为20%
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)

        for file in files:
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:
                raw = f.read()

            word_cut = jieba.cut(raw, cut_all=False)
            word_list = list(word_cut)

            data_list.append(word_list)
            class_list.append(folder)

    data_class_list = list(zip(data_list, class_list))  # zip将数据集与对应分类集打包成一个元组列表[(data_list[0],class_list[0]),()]
    random.shuffle(data_class_list)  # 将列表随机排序
    index = int(len(data_class_list) * test_size) + 1  # 划分训练集和测试集的下标
    train_list = data_class_list[index:]  # 训练集
    test_list = data_class_list[:index]  # 测试集
    train_data_list, train_class_list = zip(*train_list)  # 训练集解压缩
    test_data_list, test_class_list = zip(*test_list)

    # 统计训练集词的出现次数 即词频
    all_words_list = {}  # {word(单词):2(词频)}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_list.keys():
                all_words_list[word] += 1
            else:
                all_words_list[word] = 1

    # 根据词频倒序排序
    sorted_all_words_list = sorted(all_words_list.items(), key=operator.itemgetter(1), reverse=True)
    all_words_list, all_words_nums = zip(*sorted_all_words_list)  # 解开绑定,此时的all_word_list是排好序的迭代器
    all_words_list = list(all_words_list)
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list
# 至此得到的高频词汇表中有逗号，数字，的，了等词，要根据停用词txt去掉,且删除词频高的前N个词？？？？


"""读取txt文件内容，并去重"""
def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                words_set.add(word)
    return words_set


"""删除前N个词和停用词，得到最后的特征单词列表"""
def words_dict(all_words_list, deleteN, stopwords_set = set()):
    feature_words = []
    # 遍历第deleteN到最后一个单词，如果该单词不是数字且不在停用词表且长度为1到5，就加入到特征单词列表中
    for index in range(deleteN, len(all_words_list)):
        if not all_words_list[index].isdigit() and all_words_list[index] not in stopwords_set and 1 < len(all_words_list[index]) < 5:
            feature_words.append(all_words_list[index])
    return feature_words


"""根据feature_words即词汇表 将训练集和测试集样本向量化，1出现0未出现"""
def TextFeatures(train_data_list, test_data_list, feature_words):  # 训练集， 测试集， 单词表
    # 转化一个样本
    def text_feature(text, feature_words):  # text中单词按照是否出现在词汇表中 转化text为向量
        text = set(text)
        # features = [1 if word in text_words else 0 for word in feature_words]
        features = []
        for word in feature_words:  # 遍历词汇表单词，若在text中出现，则特征向量列表置1
            if word in text:
                features.append(1)
            else:
                features.append(0)
        return features

    # 转化所有训练集和测试集
    # train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    train_feature_list = []; test_feature_list = []
    for train_data in train_data_list:
        train_feature_list.append(text_feature(train_data,feature_words))
    for test_data in test_data_list:
        test_feature_list.append(text_feature(test_data,feature_words))
    return train_feature_list, test_feature_list


"""分类器 
    train_feature_list - 训练集向量化的特征文本
    test_feature_list - 测试集向量化的特征文本
    train_class_list - 训练集分类标签
    test_class_list - 测试集分类标签
Returns:
    test_accuracy - 分类器精度
"""
def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    # 拟合分类器
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    # fit（self，X，y [，sample_weight]）   根据X，y拟合朴素贝叶斯分类器

    test_accuracy = classifier.score(test_feature_list, test_class_list)
    # score（self，X，y [，sample_weight]）   返回给定测试数据和标签的平均精度。
    return test_accuracy


def Predict(train_feature_list, test_feature_list, train_class_list, test_class_list):
    # 拟合分类器
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    # fit（self，X，y [，sample_weight]）   根据X，y拟合朴素贝叶斯分类器
    error_count = 0
    for test_feature in test_feature_list:
        classify_result = classifier.predict(test_feature)
        index = test_feature_list.index(test_feature)
        if classify_result != test_class_list[index]:
            error_count += 1
    print('错误率：', error_count / len(test_feature_list))


if __name__ == '__main__':
    folder_path = './SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
    # print(all_words_list)
    # 至此得到的高频词汇表中有逗号，数字，的，了等词，要根据停用词txt去掉,且删除词频高的前N个词？？？？
    # 得到最后的特征单词表

    # 生成停用词表
    stopwords_set = MakeWordsSet('./stopwords_cn.txt')
    # feature_words = words_dict(all_words_list, 100, stopwords_set)
    # print(feature_words)
    # 至此已经得到过滤掉数字 停用词的单词列表，这个feature_words就是我们最终选出的用于新闻分类的特征
    # 随后，我们就可以根据feature_words，将文本向量化，然后用于训练朴素贝叶斯分类器

    """
    # 改变删除的特征词个数
    test_accuracy_list = []
    for deleteN in range(0, 1000, 20):
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list.append(test_accuracy)

    # 画图
    plt.figure()
    plt.plot(range(0, 1000, 20), test_accuracy_list)
    plt.title('特征词汇表删除个数和测试精确度关系')
    plt.xlabel('个数')
    plt.ylabel('精确度')
    # 中文不显示时：通过在程序中增加如下代码解决：
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()
    我们绘制出了deleteNs和test_accuracy的关系，这样我们就可以大致确定去掉前多少的高频词汇了。
    每次运行程序，绘制的图形可能不尽相同，我们可以通过多次测试，来决定这个deleteN的取值，
    然后确定这个参数,这样就可以顺利构建出用于新闻分类的朴素贝叶斯分类器了。
    这里选取550
    """
    test_accuracy_list = []
    feature_words = words_dict(all_words_list, 550, stopwords_set)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    test_accuracy_list.append(test_accuracy)
    # 定义一个匿名函数 函数名为ave 取c的平均值
    ave = lambda c: sum(c) / len(c)
    print(ave(test_accuracy_list))
    # print(sum(test_accuracy_list) / len(test_accuracy_list))
    # 平均精确值：0.7894736842105263













