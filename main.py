import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import os
from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():

    choice = 1

    # loss = nn.MSELoss() # 均方误差损失
    loss = nn.SmoothL1Loss() # Huber损失


    return choice, loss

def my_best():# 自行选择的超参数
    best = {'lr': 0.001, 'weight_decay': 0.01}
    return best

def data_preprocess():
    
    project_path = os.path.dirname(os.path.abspath(__file__))

    train_data = pd.read_csv(project_path + '/data/kaggle_house_pred_train.csv')
    test_data = pd.read_csv(project_path + '/data/kaggle_house_pred_test.csv')


    #忽略第一列id
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    # 若无法获得测试数据，则可根据训练数据计算均值和标准差
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    # 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
    all_features = pd.get_dummies(all_features, dummy_na=True,dtype=int)


    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to(device)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32).to(device)

    in_features = train_features.shape[1]

    return  train_data,test_data,train_features, test_features, train_labels, in_features



# 0.12
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 64),  # 输入层到隐藏层，64个节点
        nn.ReLU(),  # 非线性激活函数
        nn.Linear(64, 64),  # 隐藏层到隐藏层，64个节点
        nn.ReLU(),  # 非线性激活函数
        nn.Linear(64, 1)  # 隐藏层到输出层，1个节点
    )
    return net

# def get_net():
#     net = nn.Sequential(
#         nn.Linear(in_features, 200),
#         nn.ReLU(),
#         nn.Linear(200, 100),
#         nn.ReLU(),
#         nn.Linear(100, 1)
#     )
#     return net

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.RMSprop(net.parameters(), lr = learning_rate, weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net().to(device)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, 'f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k




# 定义目标函数
def objective(params):
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, params['lr'], params['weight_decay'], 64

    # 训练和验证的代码
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


def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net().to(device)
    train_ls, _ = train(net, train_features, train_labels, None, None,num_epochs, lr, weight_decay, batch_size)
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    
    preds = net(test_features).cpu().detach().numpy()   # 将网络应用于测试集。

    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)

    result_path = os.path.dirname(os.path.abspath(__file__)) + '/result/'
    with open(result_path + 'count.txt', 'r+', encoding='utf-8') as f:
        count = f.read()
        f.seek(0)
        f.write(str(int(count) + 1))
    path = result_path + 'submission' + count + '.csv'
    submission.to_csv(path, index=False)





if __name__ == '__main__':

    choice, loss = get_args()

    train_data,test_data,train_features, test_features, train_labels, in_features = data_preprocess()
    if choice:
        best = bayesian_optimization()
    else:
        best = my_best()
    train_and_pred(train_features, test_features, train_labels, test_data, 100, best['lr'], best['weight_decay'], 64)
    

    # 检查文件与上次有没有改变
    
    