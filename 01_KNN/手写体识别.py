import numpy as np
import operator
import os


def img2vec(filename):
    '''将32*32的样本图片转化为1*1024（1行1024列）的特征向量
        filename指图片的文件名，如3_3.txt'''

    return_vec = np.zeros((1, 1024)) ## 创建1*1024的0填充向量矩阵
    fl = open(filename)
    for i in range(32): # 读取文件的前32行，前32列     省略开头，即i从0到32
        line_str = fl.readline() #每次读一行的值
        for j in range(32):  #二层循环将32*32的值放入return_vec数组中
            return_vec[0, 32*i+j] = int(line_str[j])
    return return_vec  # 返回每个图像的向量


def handwriting_class():
    '''训练集有10*10个图片，每个图片32*32
        将训练集图片合并到100*1024的大矩阵
        每行对应一个图片'''
    handwriting_labels = []   #存储手写体类别
    training_list = os.listdir('trainingDigits')  ## 获取目录内容，返回训练集文件名列表
    m = len(training_list)
    training_vec = np.zeros((m, 1024)) #创建训练集矩阵，每行数据存储一个图像
    #将训练集图片合并到矩阵中
    for i in range(m):
        file = training_list[i]  #命名格式：1_3.txt
        filename = file.split('.')[0]  #以.切割后得到1_3
        class_label = int(filename.split('_')[0])  #以_切割得到1，即标签（类别）
        handwriting_labels.append(class_label)
        training_vec[i, :] = img2vec('trainingDigits/'+file)   ## 调用函数，每遍历一个文件就处理为二维数组中的一行


    #逐一读取测试图片，并将其分类
    test_list = os.listdir('testDigits') #创建存放测试集图片的列表
    error_count = 0.0  # 错误个数
    m_test = len(test_list)
    for i in range(m_test):
        file = test_list[i]
        filename = file.split('.')[0]
        class_label = int(filename.split('_')[0])
        #测试图片转化为矩阵
        test_vec = img2vec('testDigits/'+file)
        classify_result = classify(test_vec, training_vec, handwriting_labels, 3)
        print('\n真实值：' + str(class_label) + '预测结果为：'+ str(classify_result))
        if classify_result != class_label:
            error_count += 1.0
    print('\n出现错误个数：' + str(error_count))
    print('\n错误率为：' + str(error_count/float(m_test)))


def classify(test_data, train_data, labels, k):
    """
对未知类别属性的数据集中的每个点依次执行一下操作：

（1）计算已知类别数据集中的点与当前点之间的距离 
（2）按照距离递增次序排序 
（3）选取与当前点距离最小的k个点 
（4）确定前k个点所在类别的出现频数 
（5）返回当前k个点出现频数最高的类别作为当前点的预测分类
"""
    train_data_num = train_data.shape[0]
    diff_vec = np.tile(test_data,(train_data_num, 1)) - train_data
    sqdiss_vec = diff_vec**2
    sq_distances = sqdiss_vec.sum(axis = 1)  # array.sum(axis=1)按行累加求和，axis=0为按列累加
    distances = sq_distances**0.5

    '''计算完所有点之间的距离后，可以对数据按照从小到大的次序排序。然后，确定前k个距离
    最小元素所在的主要分类 ，输入k总是正整数；最后，将classCount字典分解为元组列表，然后
    使用程序第二行导入运算符模块的itemgetter方法，按照第二个元素的次序对元组进行排序 。
    此处的排序为逆序，即按照从最大到最小次序排序，最后返回发生频率最高的元素标签。'''
    # 计算完所有点之间的距离后，可以对数据按照从小到大的次序排序，得到在原数组中的下标列表
    sorted_dist_indicies = distances.argsort()  # 返回数组值从小到大的索引
    class_count = {}  # 创建一个字典，用于记录每个实例对应的频数
    #确定前k个距离最小元素所在的主要分类
    for i in range(k):
        vote_lable = labels[sorted_dist_indicies[i]]  # 选择k个距离最小的点，对应标签类别
        #计算类别vote_lable对应出现的次数
        # dict.get(key,x) python中字典的方法，get(key,0)从字典中获取key对应的value，字典中没有key的话返回0
        class_count[vote_lable] = class_count.get(vote_lable, 0) + 1
    # 以字典中的第二个元素（即value，此处为某类别出现的次数）逆序排序（即按照从最大到最小次序排序）
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    #返回出现次数最高的类别
    return sorted_class_count[0][0]


handwriting_class()

