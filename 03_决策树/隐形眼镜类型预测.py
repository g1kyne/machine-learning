# -*- coding:utf-8 -*-
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pydotplus  # 画图库
from sklearn.externals.six import StringIO

"""数据：
依次是age年龄、prescript症状、astigmatic是否散光、tearRate眼泪数量、  class最终的分类标签。
隐形眼镜类型包括硬材质(hard)、软材质(soft)以及不适合佩戴隐形眼镜(no lenses)

young	myope	no	reduced	no lenses
young	myope	no	normal	soft
young	myope	yes	reduced	no lenses
"""

"""sklearn.tree.DecisionTreeClassifier()函数参数说明：

sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, 
max_depth=None, min_samples_split=2, min_samples_leaf=1, 
min_weight_fraction_leaf=0.0, max_features=None, random_state=None, 
max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
class_weight=None, presort=False)

参数说明如下：

criterion：特征选择标准，可选参数，默认是gini，可以设置为entropy。gini是基尼不纯度，是将来自集合的某种结果随机应用于某一数据项的预期误差率，是一种基于统计的思想。entropy是香农熵，也就是上篇文章讲过的内容，是一种基于信息论的思想。Sklearn把gini设为默认参数，应该也是做了相应的斟酌的，精度也许更高些？ID3算法使用的是entropy，CART算法使用的则是gini。
splitter：特征划分点选择标准，可选参数，默认是best，可以设置为random。每个结点的选择策略。best参数是根据算法选择最佳的切分特征，例如gini、entropy。random随机的在部分划分点中找局部最优的划分点。默认的”best”适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐”random”。
max_features：划分时考虑的最大特征数，可选参数，默认是None。寻找最佳切分时考虑的最大特征数(n_features为总共的特征数)，有如下6种情况： 
如果max_features是整型的数，则考虑max_features个特征；
如果max_features是浮点型的数，则考虑int(max_features * n_features)个特征；
如果max_features设为auto，那么max_features = sqrt(n_features)；
如果max_features设为sqrt，那么max_featrues = sqrt(n_features)，跟auto一样；
如果max_features设为log2，那么max_features = log2(n_features)；
如果max_features设为None，那么max_features = n_features，也就是所有特征都用。
一般来说，如果样本特征数不多，比如小于50，我们用默认的”None”就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。
max_depth：决策树最大深，可选参数，默认是None。这个参数是这是树的层数的。层数的概念就是，比如在贷款的例子中，决策树的层数是2层。如果这个参数设置为None，那么决策树在建立子树的时候不会限制子树的深度。一般来说，数据少或者特征少的时候可以不管这个值。或者如果设置了min_samples_slipt参数，那么直到少于min_smaples_split个样本为止。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。
min_samples_split：内部节点再划分所需最小样本数，可选参数，默认是2。这个值限制了子树继续划分的条件。如果min_samples_split为整数，那么在切分内部结点的时候，min_samples_split作为最小的样本数，也就是说，如果样本已经少于min_samples_split个样本，则停止继续切分。如果min_samples_split为浮点数，那么min_samples_split就是一个百分比，ceil(min_samples_split * n_samples)，数是向上取整的。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。
min_weight_fraction_leaf：叶子节点最小的样本权重和，可选参数，默认是0。这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。
max_leaf_nodes：最大叶子节点数，可选参数，默认是None。通过限制最大叶子节点数，可以防止过拟合。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。
class_weight：类别权重，可选参数，默认是None，也可以字典、字典列表、balanced。指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别。类别的权重可以通过{class_label：weight}这样的格式给出，这里可以自己指定各个样本的权重，或者用balanced，如果使用balanced，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。当然，如果你的样本类别分布没有明显的偏倚，则可以不管这个参数，选择默认的None。
random_state：可选参数，默认是None。随机数种子。如果是证书，那么random_state会作为随机数生成器的随机数种子。随机数种子，如果没有设置随机数，随机出来的数与当前系统时间有关，每个时刻都是不同的。如果设置了随机数种子，那么相同随机数种子，不同时刻产生的随机数也是相同的。如果是RandomState instance，那么random_state是随机数生成器。如果为None，则随机数生成器使用np.random。
min_impurity_split：节点划分最小不纯度,可选参数，默认是1e-7。这是个阈值，这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。
presort：数据是否预排序，可选参数，默认为False，这个值是布尔值，默认是False不排序。一般来说，如果样本量少或者限制了一个深度很小的决策树，设置为true可以让划分点选择更加快，决策树建立的更加快。如果样本量太大的话，反而没有什么好处。问题是样本量少的时候，我速度本来就不慢。所以这个值一般懒得理它就可以了。
 ———————————————— 
版权声明：本文为CSDN博主「Jack-Cui」的原创文章，遵循CC 4.0 by-sa版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/c406495762/article/details/76262487
"""


# if __name__ == '__main__':
#     fr = open('lenses.txt')
#     lenses = [inst.strip().split('\t') for inst in fr.readlines()]
#     #print(lenses)  # [['young', 'myope', 'no', 'reduced', 'no lenses'], ['young',
#     lenses_target = []  # 提取每组数据的所属类别，保存在列表中
#     for lense in lenses:
#         lenses_target.append(lense[-1])
#
#     lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 特征标签
#     clf = tree.DecisionTreeClassifier()  # 决策树分类器
#     lenses = clf.fit(lenses, lensesLabels)  # 拟合模型

if __name__ == '__main__':
    with open('lenses.txt', 'r') as fr:                                        # 加载文件
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]        # 处理文件
    lenses_target = []                                                        # 提取每组数据的分类类别标签，保存在列表里
    for each in lenses:
        lenses_target.append(each[-1])

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']    # 特征标签
    # lenses_list = []    # 保存每个特征标签对应的数据的临时列表  ['young', 'young', 'young', 'young', 'young', 'young', 'young'...]
    lenses_dict = {}   # 保存lenses数据的字典，用于生成pandas  {'age': ['young', 'young', 'young', 'young', 'young', 'young', 'young'...]'prescript': ['myope', 'myope', 'myope', 'my..]}
    for lenseLabel in lensesLabels:  # 遍历每个特征标签
        lenses_list = []  # 保存每个特征标签对应的数据的临时列表  ['young', 'young', 'young', 'young', 'young', 'young', 'young'...]
        for lense in lenses:
            lenses_list.append(lense[lensesLabels.index(lenseLabel)])  # 将age对应的下标 对应在数据集中的取值存入list中
        lenses_dict[lenseLabel] = lenses_list  # 算完age的所有取值后，加入到age对应的value中

    lenses_pd = pd.DataFrame(lenses_dict)  # 将字典转换为pandas数据（为了对string类型的数据序列化）
    # 生成pandas.DataFrame
    #     age      prescript     astigmatic    tearRate
    # 0  young     myope         no           reduced
    # print(lenses_pd)

    #序列化
    le = LabelEncoder()  # 创建对象，用于序列化   LabelEncoder:将字符串转换为增量值
    for col in lenses_pd.columns:  # 每一列序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])  # fit后进行标准化
    # print(lenses_pd)

# 序列化前
    # 生成pandas.DataFrame
    #     age      prescript     astigmatic    tearRate
    # 0  young     myope         no           reduced
# 序列化后
    #     age      prescript     astigmatic    tearRate
    # 0   2        0             1              1

    # 画图
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    dot_data = StringIO()  # StringIO顾名思义就是在内存中读写str
    tree.export_graphviz(clf, out_file=dot_data,   # 绘制决策树
                         feature_names=lenses_pd.keys(),
                         class_names=lenses_target,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  # getvalue()：返回对象s中的所有数据
    graph.write_pdf('tree.pdf')


# 预测
print(clf.predict([[1, 1, 1, 0]]))











