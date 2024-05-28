import numpy as np
import pandas as pd
import torch
from torch import nn
import os
from hyperopt import fmin, tpe, hp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自行预定义
def get_args():

    choice = 1

    # loss = nn.MSELoss() # 均方误差损失
    # loss = torch.sqrt(nn.MSELoss()) # 均方根误差损失
    # loss = nn.L1Loss() # L1损失
    loss = nn.SmoothL1Loss() # Huber损失

    my_optimizer = torch.optim.RMSprop
    
    k=5
    num_epochs=500
    batch_size=64
    patience=30

    # 0.11979 submission13
    # num1 = 127
    # num2 = 275

    # 0.11926 submission15 
    # {'lr': 0.010000932608412779, 'num1': 35, 'num2': 132, 'weight_decay': 0.2040251030562371}

    #  submission16     MSELoss      Adam
    # {'lr': 0.7599159441457433, 'num1': 251, 'num2': 58, 'weight_decay': 0.021198893106332467} 训练log rmse：0.042248

    # submission17          L1Loss      Adam
    # {'lr': 0.02282028181603883, 'num1': 191, 'num2': 11, 'weight_decay': 0.0357673907830876} 训练log rmse：0.245027

    # submission18          L1Loss      RMSprop
    # submission19          MSELoss      RMSprop
    # submission20          SmoothL1Loss      RMSprop
    # submission21          {'lr': 0.00832718465955878, 'num1': 210, 'num2': 640, 'weight_decay': 0.12045844009875302}
    # submission22          dropout
    # submission23          dropout random?
    # submission24          num123,Early stop:patience=10       {'lr': 0.03056633337169984, 'num1': 116, 'num2': 530, 'num3': 278, 'weight_decay': 0.35030074672470984} 训练log rmse：0.037608
    # submission25          patience=20             {'lr': 0.007324901971771072, 'num1': 425, 'num2': 517, 'num3': 6, 'weight_decay': 0.026031871210853732}
    # submission26          patience=30             
    
    return choice, loss, my_optimizer, k, num_epochs, batch_size, patience

# 自行选择的超参数
def my_best():
    best = {'lr': 0.008854963202072644, 'weight_decay': 0.045949699090015866, 'num1': 159, 'num2': 300}
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

    print(train_features.shape)
    return  train_data,test_data,train_features, test_features, train_labels, in_features



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

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size, num1, num2,num3, patience):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = my_optimizer(net.parameters(), lr = learning_rate, weight_decay = weight_decay)
    best_test_loss = float('inf')
    epochs_no_improve = 0

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


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, num1, num2,num3, patience):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(in_features, num1, num2,num3).to(device)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size, num1, num2,num3, patience)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, 'f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k



# 贝叶斯优化
# 定义参数空间
def bayesian_optimization():
    space = {
        'lr': hp.loguniform('lr', -5, 0),
        'weight_decay': hp.loguniform('weight_decay', -5, 0),
        'num1': hp.choice('num1', range(2, 700)),
        'num2': hp.choice('num2', range(2, 700)),
        'num3': hp.choice('num3', range(2, 700))
    }

    # 运行优化
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=100)

    print(best)
    return best
# 定义目标函数
def objective(params):
    lr, weight_decay, num1, num2, num3 =  params['lr'], params['weight_decay'],  params['num1'], params['num2'],params['num3']

    # 训练和验证的代码
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size, num1, num2,num3, patience)
    # 返回验证误差
    return valid_l



def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size,num1,num2,num3,patience):
    net = get_net(in_features,num1,num2,num3).to(device)
    train_ls, _ = train(net, train_features, train_labels, None, None,num_epochs, lr, weight_decay, batch_size,num1,num2,num3,patience)
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    
    preds = net(test_features).cpu().detach().numpy()   # 将网络应用于测试集。

    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)

    result_path = os.path.dirname(os.path.abspath(__file__)) + '\\result\\'
    with open(result_path + 'count.txt', 'r+', encoding='utf-8') as f:
        count = (f.readlines()[-1])
        
        path = result_path + 'submission' + count + '.csv'
        submission.to_csv(path, index=False)
        f.write(' ' + str(float(train_ls[-1])) + '\n' + str(int(count) + 1))




if __name__ == '__main__':

    choice, loss , my_optimizer, k, num_epochs, batch_size, patience = get_args()

    train_data,test_data,train_features, test_features, train_labels, in_features = data_preprocess()
    if choice:
        best = bayesian_optimization()
    else:
        best = my_best()
    train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, best['lr'], best['weight_decay'], batch_size, best['num1'], best['num2'],best['num3'],patience)
    

    # 检查文件与上次有没有改变
    
    