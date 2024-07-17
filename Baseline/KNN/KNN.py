import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import operator 
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
import argparse
import os

path = os.path.abspath(os.path.join(os.getcwd(), ".."))

# 加载数据
Xtrain = np.load(path + '/DDoS_Detection/CICDDoS2019/train.npz',allow_pickle=True)
Xtest = np.load(path + '/DDoS_Detection/CICDDoS2019/test.npz',allow_pickle=True)

testLabel = Xtest['y']

# 归一化(Z-score归一化)
trainDataset = (Xtrain['x'] - np.mean(Xtrain['x'], axis=0)) / np.std(Xtrain['x'], axis=0)
testDataset = (Xtest['x'] - np.mean(Xtest['x'], axis=0)) / np.std(Xtest['x'], axis=0)

# 使用GPU计算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainDataset = torch.tensor(trainDataset,device=device)
testDataset = torch.tensor(testDataset,device=device)
trainLabel = torch.tensor(Xtrain['y'].astype(float),device=device)


def Distance(trainDataset,testData):
    """
    trainDataset: 训练集
    testData: 每一条测试数据
    """
    DistanceList = torch.sqrt(torch.sum((trainDataset - testData) ** 2, dim=1)) # 涉及广播机制
    return DistanceList            

def Classifer(DistanceList, trainLabel, k):
    """
    DistanceList: 距离列表
    trainLabel: 训练集标签
    k: 选取的k值
    """
    _, index = torch.topk(DistanceList, k, largest=False)
    NeighborLabel = trainLabel[index]
    NeighborLabel, counts = torch.unique(NeighborLabel, return_counts=True)
    PreLabel = NeighborLabel[torch.argmax(counts)].item()
    return PreLabel # 预测的类别

def metric(testLabel, PredictionList):  # 模型评估  
    """
    testLabel: 测试集标签
    PredictionList: 预测标签
    """
    accuracy = accuracy_score(np.array(testLabel).astype(int),np.array(PredictionList).astype(int))
    precision = precision_score(np.array(testLabel).astype(int),np.array(PredictionList).astype(int))
    recall = recall_score(np.array(testLabel).astype(int),np.array(PredictionList).astype(int))
    f1 = f1_score(np.array(testLabel).astype(int),np.array(PredictionList).astype(int))
    print('accuracy:',accuracy)
    print('precision:',precision)
    print('recall:',recall)
    print('f1:',f1)


if __name__ == '__main__':
    
    parse = argparse.ArgumentParser()
    parse.add_argument('--K',type=int, default=100, help='The top K')
    args = parse.parse_args()
    
    k = args.K
    # 遍历测试集(计算每一个测试集样本)
    PredictionList = []
    for i in tqdm(range(len(testDataset))):
        DistanceList = Distance(trainDataset,testDataset[i])
        Prediction = Classifer(DistanceList, trainLabel, k)
        PredictionList.append(Prediction)
    
    metric(testLabel, PredictionList)
