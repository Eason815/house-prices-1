# 房价预测

## 项目源

https://www.kaggle.com/c/house-prices-advanced-regression-techniques



## 引言

让一个购房者描述他们梦想中的房子，他们可能不会从地下室天花板的高度或者东西向铁路的接近开始。但是，这个游乐场竞赛的数据集证明，比起卧室的数量或白色尖桩篱笆，更多的影响价格谈判。

有79个解释性变量描述(几乎)在爱荷华州埃姆斯的住宅的每一个方面，这个比赛挑战你预测每个家庭的最终价格。

## 数据集

该数据集由 Bart de Cock 在2011年(DeCock，2011)收集，涵盖了2006-2010年间内务部 Ames 的房价。它比著名的波士顿住房数据集哈里森和鲁宾菲尔德(1978)大得多，拥有更多的例子和更多的功能。

**使用说明及下载**：https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data


## 评价

截止此次Commit 

最高在Kaggle得分为0.11979 排名约为145/4768






## 思路&优化点

### 损失函数

常用的损失函数有均方误差（Mean Squared Error，MSE）和平均绝对误差（Mean Absolute Error，MAE），以及Huber损失（也称为Smooth Mean Absolute Error）。

1. **均方误差（MSE）**：最常用的回归问题损失函数。它计算预测值和真实值之间的平方差。但是，MSE对于异常值非常敏感，因为它会将每个误差平方。

```python
loss = nn.MSELoss()
```

2. **平均绝对误差（MAE）**：MAE是预测值和真实值之间差的绝对值的平均。与MSE相比，MAE对异常值不那么敏感，但它不是可微的，这可能会影响优化算法的性能。

```python
loss = nn.L1Loss()
```

3. **Huber损失**：Huber损失是MSE和MAE的折衷，它在误差较小的时候表现得像MSE，在误差较大的时候表现得像MAE。这使得它对于异常值不那么敏感。

```python
loss = nn.SmoothL1Loss()
```
大多数情况下

如果数据中存在很多异常值，适合选择MAE或Huber损失。

如果数据中异常值不多，MSE适合。

### 优化器

这里考虑以下几种优化器：

1. **随机梯度下降（SGD）**：这是最基本的优化器，但在某些情况下可能需要较长的时间才能收敛。

```python
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
```

2. **Adam**：Adam优化器结合了RMSProp和Momentum的优点，通常表现得很好，是一种常用的优化器。

```python
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
```

3. **RMSprop**：RMSprop优化器是一种自适应学习率的优化器，可以处理非平稳目标函数和在线和非平稳设置。

```python
optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
```
大多数情况下

如果数据是稀疏的，适合选择Adam或RMSprop。

如果有关深度学习问题，适合选择Adam。

### 正则化

实现正则化以防止过拟合：

1. **权重衰减**：在优化器中设置`weight_decay`参数可以实现L2正则化，这是一种常见的权重衰减方法。如：

```python
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.01)
```

2. **Dropout**：在模型中添加`nn.Dropout`层可以在训练过程中随机关闭一部分神经元，这也可以防止过拟合。如：

```python
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Dropout(0.5),  # 添加Dropout层，丢弃率为0.5
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(0.5),  # 添加Dropout层，丢弃率为0.5
        nn.Linear(64, 1)
    )
    return net
```

在此项目中实战效果不佳，后续没有再考虑使用。


### 超参数选择(关键点)

人手选择？抽卡？

那肯定是使用以下常用的策略：


1. **网格搜索**：为每个超参数定义一个候选值的列表，然后尝试所有可能的组合。
2. **随机搜索**：当有很多超参数时，网格搜索非常耗时。随机搜索是一种替代策略。
3. **贝叶斯优化**：使用贝叶斯推理来预测哪一组超参数可能会给出最好的性能。然后它在这个预测的基础上选择下一组要尝试的超参数。
4. **学习率衰减**：对于学习率，一种常见的策略是开始时使用较大的值，然后在训练过程中逐渐减小。这可以通过设置学习率调度器来实现。

```贝叶斯优化```是一种用于找到最优化问题的全局最优解的方法，特别适合在高维度和非凸情况下寻找超参数的最优解。它通过构建目标函数的概率模型，然后使用这个模型来选择下一次的查询点。




### 网络模型


考虑以下几种网络模型：

1. **线性回归（Linear Regression）**：这是最基本的回归模型，适用于特征和目标之间存在线性关系的情况。但如果特征和目标之间的关系是非线性的，线性回归的表现可能就不会很好。

2. **多层感知机（Multilayer Perceptron，MLP）**：MLP是一种简单的神经网络，由一个输入层、一个或多个隐藏层和一个输出层组成。MLP可以模拟非线性关系。

3. **深度神经网络（Deep Neural Network，DNN）**：DNN是一种有多个隐藏层的神经网络。DNN可以模拟更复杂的非线性关系，但也更容易过拟合。

基本上都是选择MLP或者DNN。

这里的实验开始只是使用简单的线性回归模型

```python
def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net
```

我们可以从这里着手进行改进优化，如：

```python
def get_net(in_features,num1,num2):
    net = nn.Sequential(
        nn.Linear(in_features, num1),
        nn.ReLU(),
        nn.Linear(num1, num2),
        nn.ReLU(),
        nn.Linear(num2, 1)
    )
    return net
```

## 实现过程

使用d2l课本源代码 

在Kaggle得分为0.16713 排名约为3516/4768



### 尝试1

改进损失函数


~~loss = nn.MSELoss() # 均方误差损失~~

    loss = nn.SmoothL1Loss() # Huber损失


改进优化器

~~optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay = weight_decay) # Adam优化器~~


    optimizer = torch.optim.RMSprop(net.parameters(), lr = learning_rate, weight_decay = weight_decay) #RMSprop优化器

(后面经多次验证，个人认为最佳组合)

在Kaggle得分为0.16703 排名约为3460/4768


### 尝试2

改进网络模型使用DNN/MLP，设置三层线性层

```python
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 64),  # 输入层到隐藏层，64个节点
        nn.ReLU(),  # 非线性激活函数
        nn.Linear(64, 64),  # 隐藏层到隐藏层，64个节点
        nn.ReLU(),  # 非线性激活函数
        nn.Linear(64, 1)  # 隐藏层到输出层，1个节点
    )
    return net
```

改进选择参数方式，使用贝叶斯优化来确定超参数(学习率，L2正则的权重衰减)
```python
from hyperopt import fmin, tpe, hp

# 定义目标函数
def objective(params):
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, params['lr'], params['weight_decay'], 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size) # 训练和验证
    return valid_l  # 返回验证误差

# 贝叶斯优化
def bayesian_optimization():
    # 定义参数空间
    space = {'lr': hp.loguniform('lr', -5, 0),'weight_decay': hp.loguniform('weight_decay', -5, 0),}
    # 运行优化
    best = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=100)
    return best
```

贝叶斯优化选择了合适的超参数后效果很明显

在Kaggle得分为0.12391 排名约为446/4768


### 尝试3

对网络模型的改进仍抱有一丝希望

个人观点是对于一个比较简单的房价预测问题，没有必要使用过于复杂模型，大材小用


于是跑去尝试使用穷举法对main运行1000次，对优质网络模型进行选拔hh


```python
import subprocess

for i in range(1000):
    subprocess.call(["python", "main.py"])

------------------------------------------------------------

num1 = random.randint(10, 300)
num2 = random.randint(10, 300)
def get_net(in_features,num1,num2):

    net = nn.Sequential(
        nn.Linear(in_features, num1),
        nn.ReLU(),
        nn.Linear(num1, num2),
        nn.ReLU(),
        nn.Linear(num2, 1)
    )
    return net
```


网络模型具有一定随机性，依靠此特性多次训练并抽卡

在Kaggle得分为0.12113 排名约为216/4768


### 尝试4

使用早停法确定合适epoch
(效果非常不佳，已禁用)
```python
for epoch in range(num_epochs):
    for X, y in train_iter:
        optimizer.zero_grad()
        l = loss(net(X), y)
        l.backward()
        optimizer.step()
    train_loss = log_rmse(net, train_features, train_labels)
    train_ls.append(train_loss)
    if test_labels is not None:
        test_loss = log_rmse(net, test_features, test_labels)
        test_ls.append(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping! epoch now is'+ str(epoch))
                return train_ls, test_ls
```

想到既然能用贝叶斯选择参数，那就顺便让他决定网络模型，于是改进尝试3

```python
# 定义参数空间
def bayesian_optimization():
    space = {
        'lr': hp.loguniform('lr', -5, 0),
        'weight_decay': hp.loguniform('weight_decay', -5, 0),
        'num1': hp.choice('num1', range(2, 700)),
        'num2': hp.choice('num2', range(2, 700))
    }
    # 运行优化
    best = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=100)
    return best
# 定义目标函数
def objective(params):
    lr, weight_decay, num1, num2 =  params['lr'], params['weight_decay'],  params['num1'], params['num2']
    # 训练和验证的代码
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size, num1, num2, patience)
    # 返回验证误差
    return valid_l
```

在Kaggle得分为0.11926 排名约为115/4788

### 尝试5

尝试再加一层为4层
(效果不佳，回滚使用3层)
```python
def get_net(in_features,num1,num2,num3):
    net = nn.Sequential(
        nn.Linear(in_features, num1),
        nn.ReLU(),
        nn.Linear(num1, num2),
        nn.ReLU(),
        nn.Linear(num2, num3),
        nn.ReLU(),
        nn.Linear(num3, 1)
    )
    return net
```
### 尝试6

貌似已经优化到将近极致，最后再对数据进行处理一下

自行尝试检测和剔除异常值

详见`dealdata.ipynb`文件

首先，通过计算相关性挑出几个值高的重要指标

(主要是要找几个样例进行可视化，看看剔除效果，而这些又比较重要)

其次，选择算法

在`dealdata.ipynb`里，个人研究并尝试以下6种
    
- Z-scores
- 箱线图/IQR
- 马氏距离
- KMeans聚类
- DBSCAN聚类
- LOF

并作了可视化分析



## 总结

对于机器学习问题，抓住主要部分

1. 数据

- 这里课本数据预处理上较为完善，包括：
    
    * 标准化数值特征(标准化:减去均值然后除以标准差)
    * 填充缺失值
    * 创建虚拟变量
    * 划分训练集和测试集

- 对train_data剔除异常值
    * LOF算法

2. 模型

- 这里课本使用的简单线性回归模型。


- 改进使用
    * MLP/DNN模型

3. 目标函数

目标函数通常就是指损失函数。

- 均方误差（MSE）
- Huber损失



4. 调参

还是在这里对项目进行突破了

全局优化：

- 贝叶斯优化

局部优化：

- Adam
- RMSprop

## 参考文献

https://d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html

在课本项目基础上优化改进，在Kaggle排行进行验证

## 署名

22软工lxx