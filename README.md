# 机器学习实现信用卡交易欺诈检测（利用Intel-oneAPI技术）

### 团队名称：
天枢信安小组

### 团队成员：
陈冠旭、金起正，洪淳宇

### 一、问题背景陈述

2021 年，与信用卡欺诈相关的损失超过 120 亿美元，同比增长近 11%。就重大财务损失、信任和信誉而言，这是银行、客户和商户面临的一个令人担忧的问题。
电子商务相关欺诈一直在以约 13% 的复合年增长率 (CAGR) 增加。由于欺诈性信用卡交易急剧增加，在交易时检测欺诈行为对于帮助消费者和银行非常重要。机器学习可以通过训练信用卡交易模型，然后使用这些模型更快、更准确地检测欺诈交易，在预测欺诈方面发挥至关重要的作用。


### 二、项目简介

本项目希望通过在给定的数据集内按照给定的信息对欺诈操作进行检测，对一次操作，有欺诈和非欺诈两种类别，是典型的二分类问题。使用的数据集包含欧洲持卡人于 2013 年 9 月通过信用卡进行的交易信息。这些交易在两天内发生，284827笔交易中存在492次欺诈，正类仅占所有交易的0.172%，数据集极不平衡。

### 三、项目使用框架与技术

1. scikit-learn机器学习库
2. 英特尔OneAPI机器学习加速平台及其第三方库

### 四、数据预处理

####  1. 我们定义了一系列的处理函数，主要功能如下：

- readData，从指定路径读取数据
- preProcessData，对数据进行预处理，主要包括一下两个步骤
  - 对Amount进行归一化，避免因数据波动范围过大，对模型效果产生影响
  - 删除time列，我们认为时间与是否为欺诈交易没有直接逻辑关系，所以删除此列消除time对预测结果的影响
- splitData
  - 将传入的data分割为特征集合x与标签集合y
- getTrainTestSet
  - 将特征集合与标签集合分隔为训练集与测试集
- showClassDistribution
  - 展示标签集合内的样本分布
- showXDistribution
  - 展示特征集合内的样本分布

#### 2. 数据集构造过程如下：

- 读取数据
- 清洗数据，对Amount进行归一化同时删除time列
- 分割特征集合与标签集合
- 对数据进行重新采样，此处采用欠采样的方式
- 按照4:1的比例分割训练集与测试集

#### 3. 数据集欠采样处理

因为实验数据中class=0与class=1的样本数目分布极不均衡，无疑选择欠采样的处理方式获得数据集。我们在实验中发现在进行欠采样处理时，保持class=1的样本数目不变，增大class=0的样本数目，模型在原数据集上的整体预测效果更好，因此我们在`selectSampleRatio.py`文件中该比例进行了探究，使用逻辑回归模型的f1分数作为标准，结果如下：

![f1_and_accuracy_with_dataset_percentage](.\images\f1_and_accuracy_with_dataset_percentage.png)

我们发现在该比例小于50时，模型表现会随着比例增加而较快增长，在比例到达100附近后达到饱和，曲线接近平稳。

我们最终选择使用100的规模比（注意：这在带来模型训练效果增强的同时也会造成训练时间开销的巨大增长），另附上我们重新采样所构建的数据集的特征分布与标签分布：

<img src=".\images\class_distri.jpg" alt="class_distri" style="zoom:50%;" />

![feature_distri1](.\images\feature_distri1.jpg)

![feature_distri2](.\images\feature_distri2.jpg)

分析原因如下：

- 在原数据集中，class=1的样本数目接近500条，如果按照1:1的比例进行采样，所得数据集数据量过小导致模型训练效果不好

### 五、训练模型

#### 1. 综合使用七种模型进行对比分析

我们在实验中使用决策树、随机森林、逻辑回归、支持向量机SVM、逻辑回归、朴素贝叶斯、k-邻近算法分类器KNN、梯度提升树XGBoost等七种模型进行分类，充分比较各种模型的分类效果并取最优。

#### 2. 使用格搜索进行参数调优

在机器学习中，调参是一件非常棘手的问题，我们使用格搜索的方式进行模型参数选择，为每一种模型都定义了一个函数modelGS以实现该功能。

以逻辑回归模型lr的参数调优函数`lrGS`为例:

```python
def lrGS(lr,x_train,y_train):
    param_grid = {
        'max_iter': [100, 200,300]
    }

    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1')
    grid_search.fit(x_train, y_train)

    return grid_search.best_params_
```

该函数使用格搜索遍历所有的参数组合，并返回一个最优组合。

```python
lr_params = lrGS(LogisticRegression(random_state=42), x_train, y_train)
lr = LogisticRegression(**lr_params)
```

通过上述语句，生成一个使用最优参数的逻辑回归分类器。

然而，遗憾的是，随着参数数目与每个参数选择范围大小的增长，格搜索会带来训练时间的巨大开销，而我们在前面已经选择了较大的样本集合，受限于时间限制，我们最终选择适当减少格搜索的参数种类与参数范围。

### 六、测试效果

#### 1. 测试方式

我们封装了通用的测试函数`uniformTest`，打印并返回对应的F1分数、准确率和推理时间。

#### 2. 测试标准

考虑对样本标签分布极不均匀，在准确率与f1分数之间更关注于f1分数。

#### 3. 测试集

我们在这里采用了两个测试集：

- 在划分训练集与测试集时所构建的测试集合
- 原始给定数据集

我们发现在所构建的测试集中可以达到相当高的预测准确度，因此为了体现模型的真实效果，更关注于后者的准确性。

### 七、可视化测试结果展示

##### 1. 测试集

F1分数

<img src=".\images\f1_score_over_testset.png" alt="f1_score_over_testset" style="zoom:50%;" />

##### 2. 原数据集

F1分数

<img src=".\images\f1_score_over_origin_dataset.png" alt="f1_score_over_origin_dataset" style="zoom:50%;" />

综合两个数据集上的表现而言，xgboost和随机森林无疑是众多模型中表现最好的，在测试集合原数据集上的F1分数都达到了0.9以上。但是后面在讨论时间问题是会发现随机森林算法在加速前会使用最多的时间，综合考虑xgboost在该问题上分类表现最好。

另附上所有模型在源数据集上的混淆矩阵

<img src=".\images\tree_cm_plot.png" alt="tree_cm_plot" style="zoom:50%;" />

<img src=".\images\rf_cm_plot.png" alt="rf_cm_plot" style="zoom:50%;" />

<img src=".\images\svm_cm_plot.png" alt="svm_cm_plot" style="zoom:50%;" />

<img src=".\images\lr_cm_plot.png" alt="lr_cm_plot" style="zoom:50%;" />

<img src=".\images\mnb_cm_plot.png" alt="mnb_cm_plot" style="zoom:50%;" />

<img src=".\images\knn_cm_plot.png" alt="knn_cm_plot" style="zoom:50%;" />

<img src=".\images\xgb_cm_plot.png" alt="xgb_cm_plot" style="zoom:50%;" />

### 八、利用OneAPI加速实验

我们注意到在前面的实验中随着数据集规模的增加、随着调优需求的增长，时间成为模型训练的关键问题，于是我们在第五部分进一步利用OneAPI来加速该过程。对使用OneAPI前后的训练时间和推理时间进行比较如下：

##### 训练时间

<img src=".\images\training_time.png" alt="training_time" style="zoom:50%;" />

使用OneAPI后

<img src=".\images\intel_training_time.png" alt="intel_training_time" style="zoom:50%;" />

##### 推理时间

<img src=".\images\prediction_time_over_testset.png" alt="prediction_time_over_testset" style="zoom:50%;" />

<img src=".\images\prediction_time_over_origin_dataset.png" alt="prediction_time_over_origin_dataset" style="zoom:50%;" />

使用oneAPI后

<img src=".\images\intel_prediction_time_over_testset.png" alt="intel_prediction_time_over_testset" style="zoom:50%;" />

<img src=".\images\intel_prediction_time_over_origin_dataset.png" alt="intel_prediction_time_over_origin_dataset" style="zoom:50%;" />

我们惊讶地发现，最显著地使用OneAPI库加速后，随机森林训练速度加速数十倍，KNN推理速度加速近七倍，OneAPI实现了极为惊人的加速比，这为我们进一步增强模型能力提供了支持。

比如，我们可以使用OneAPI实现我们之前为减少时间开支而注释掉的格搜索部分

### 九、项目总结

在整个实验中，我们惊讶地发现OneAPI库可以将随机森林的训练时间加快数十倍，KNN的推理速度加速近七倍，如此优秀的加速比为我们解决数据规模大和参数调优带来的时间开销问题提供了支持，在AI迅速发展的今天，可以大幅消减对应的时间与硬件资源开销。

对OneAPI加速方式的一点思考：注意到随机森林是加速比最大的一项和它的特殊结构，我们认为对并行能力的挖掘在OneAPI的加速上发挥了重要的作用。

还经查询得到OneAPI的加速原理：

> oneAPI的优势在于充分发挥了异构计算的潜力，通过将计算任务分配给最适合执行的硬件，使得算法在训练和测试集上的运行速度得到显著提升。这种高效性能的实现为大规模数据处理和机器学习等领域提供了更快速、可扩展的解决方案，同时在不同硬件平台上实现了可移植性。 

### 十、参考内容

[oneapi-src/credit-card-fraud-detection: AI Starter Kit for Credit Card Fraud Detection model using Intel® Extension for Scikit-learn* (github.com)](https://github.com/oneapi-src/credit-card-fraud-detection)

