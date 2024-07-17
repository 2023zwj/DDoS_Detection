import torch 
import matplotlib.pyplot as plt 
import random  
import pandas as pd
import numpy as np 
import torch.nn as nn  
import torch.utils.data as Data 
from torch.nn import init  
import torch.optim as optim  
from LogisticRegression import LogisticRegression 
from conf import parse
import os 

args = parse.parse_args()

# 全局变量
batch_size = args.batch_size
num_epochs = args.epochs
num_inputs = args.feature_dis # 特征维度

path = os.path.abspath(os.path.join(os.getcwd(), ".."))

# 加载数据
Xtrain = np.load(path + '/DDoS_Detection/CICDDoS2019/train.npz',allow_pickle=True)
Xtest = np.load(path + '/DDoS_Detection/CICDDoS2019/test.npz',allow_pickle=True)

# 归一化
train_x = (Xtrain['x'] - np.mean(Xtrain['x'], axis=0)) / np.std(Xtrain['x'], axis=0)
test_x = (Xtest['x'] - np.mean(Xtest['x'], axis=0)) / np.std(Xtest['x'], axis=0)

train_dataset = Data.TensorDataset(torch.tensor(train_x.astype(float),dtype=torch.float32),torch.tensor(Xtrain['y'].astype(float),dtype=torch.float32)) 
test_dataset = Data.TensorDataset(torch.tensor(test_x.astype(float),dtype=torch.float32),torch.tensor(Xtest['y'].astype(float),dtype=torch.float32)) 

train_data_iter = Data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers=0,
)
test_data_iter = Data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers=0,
)


model = LogisticRegression(num_inputs)  
loss = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr = args.lr)


def train(data_iter,net):
    cnt = 0
    train_Loss = 0
    for X,y in data_iter:
        X, y = X.float(), y.float()
        output = net(X) 
        Loss = loss(output,y.view(-1,1))
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        train_Loss = train_Loss + Loss.item()   
        cnt = cnt + 1
    return train_Loss/cnt

def test(data_iter,net):
    with torch.no_grad():
        cnt = 0
        test_Loss = 0
        for X,y in data_iter:
            output = net(X) 
            Loss = loss(output,y.view(-1,1))
            test_Loss = test_Loss + Loss   
            cnt = cnt + 1
    return test_Loss/cnt

if __name__ == "__main__":
    for epoch in range(1,num_epochs + 1):
        train_loss = train(train_data_iter,model) 
        test_loss = test(test_data_iter,model) 
        print("epoch: %d  train_loss: %f test_loss: %f" %(epoch,train_loss,test_loss))
    
    # 评估指标
    y_pre = model.Pre(torch.tensor(train_x).float())
    y_pre = y_pre.reshape(-1) # 改变形状
    accuracy,precision,recall,f1 = model.metric(torch.tensor(Xtrain['y'].astype(float)).float(), y_pre.detach())
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)