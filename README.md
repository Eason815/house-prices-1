# 房价预测

## 项目源

https://www.kaggle.com/c/house-prices-advanced-regression-techniques



## 引言

让一个购房者描述他们梦想中的房子，他们可能不会从地下室天花板的高度或者东西向铁路的接近开始。但是，这个游乐场竞赛的数据集证明，比起卧室的数量或白色尖桩篱笆，更多的影响价格谈判。

有79个解释性变量描述(几乎)在爱荷华州埃姆斯的住宅的每一个方面，这个比赛挑战你预测每个家庭的最终价格。

## 数据集

该数据集由 Bart de Cock 在2011年(DeCock，2011)收集，涵盖了2006-2010年间内务部 Ames 的房价。它比著名的波士顿住房数据集哈里森和鲁宾菲尔德(1978)大得多，拥有更多的例子和更多的功能。

**使用说明及下载**：https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data




## 实现过程

使用d2l课本源代码 在Kaggle得分为0.16 

### 尝试1



在处理房价预测这类回归问题时，以下是一些常用的深度学习模型：

1. **多层感知机（MLP）**：这是最简单的深度学习模型，由多个全连接层组成。尽管简单，但在许多任务中表现良好。

```python
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    return net
```

2. **卷积神经网络（CNN）**：虽然CNN主要用于图像处理，但在某些情况下，也可以用于结构化数据。例如，你可以将数据重新排列成二维形式，并使用CNN处理。

3. **自编码器（Autoencoder）**：自编码器是一种无监督的深度学习模型，可以用于特征提取。你可以训练一个自编码器来学习数据的低维表示，然后使用这个表示进行预测。

4. **变分自编码器（VAE）或生成对抗网络（GAN）**：这些是更复杂的无监督学习模型，可以用于学习数据的复杂分布。然而，它们通常更难训练，可能不适合初学者。

请注意，选择哪种模型取决于你的具体任务和数据。在选择模型时，你应该考虑你的数据类型（例如，你是否有图像或文本数据），你的任务（例如，你是否正在进行分类或回归），以及你的计算资源。

## 优化点

### 贝叶斯优化

使用贝叶斯推理来预测哪一组超参数可能会给出最好的性能。然后它在这个预测的基础上选择下一组要尝试的超参数。

```python
from hyperopt import fmin, tpe, hp

# 定义目标函数
def objective(params):
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, params['lr'], params['weight_decay'], 64
    # 训练和验证
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    # 返回验证误差
    return valid_l

# 贝叶斯优化
def bayesian_optimization():
    # 定义参数空间
    space = {
        'lr': hp.loguniform('lr', -5, 0),
        'weight_decay': hp.loguniform('weight_decay', -5, 0),
    }

    # 运行优化
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=100)

    print(best)
    return best
```

### 