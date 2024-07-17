import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os

path = os.path.abspath(os.path.join(os.getcwd(), ".."))

# 计算先验概率
def PriorPre(Xtrain):
    """
    Xtrain: 训练集
    """
    POS = np.sum(Xtrain['y']) / len(Xtrain['y']) # 正样本先验概率
    NEG = 1 - POS # 负样本先验概率
    return POS, NEG

# 分离正负样本
def DepartPN(Xtrain):
    """
    Xtrain: 训练集
    """
    POStrain = []
    NEGtrain = []
    
    POStrain = Xtrain['x'][Xtrain['y'] == 0]  # 正样本
    NEGtrain = Xtrain['x'][Xtrain['y'] == 1]  # 负样本

    return POStrain, NEGtrain

# 均值
def Average(feature):
    """
    feature: 待求均值的特征矩阵
    """
    feature = np.array(feature)
    avg = np.sum(feature,axis = 0) / len(feature)
    return avg

# 方差
def Variance(feature):
    """
    feature: 待求均值的特征矩阵
    """
    avg = Average(feature)
    feature = np.array(feature)
    Med = pow((feature - avg),2) # 涉及广播机制
    Var = np.sum(Med, axis = 0) / len(feature) # 方差结果
    return Var 

def train(feature, testData):
    """
    feature: 训练集特征矩阵
    testData: 测试集数据
    """

    avg = Average(feature) # 均值
    var = Variance(feature)  # 方差
  
    # 计算正态分布概率密度函数
    Prediction = np.prod((1 / (np.sqrt(2 * np.pi) * var)) * np.exp(- (pow((testData - avg), 2)) / (2 * pow(var,2))), axis = 1)
    return Prediction

def Classifer(Xtrain, testData):
    """
    Xtrain: 训练集
    testData: 测试数据
    """
    POS, NEG = PriorPre(Xtrain)
    POStrain, NEGtrain = DepartPN(Xtrain)
    PosPre = train(POStrain,testData) * POS
    NegPre = train(NEGtrain,testData) * NEG

    Prediction = (PosPre < NegPre).astype(int)
    return Prediction
    
def test(Xtrain, Xtest):
    """ 
    Xtrain: 训练集
    Xtest: 测试集
    """
    PreList = Classifer(Xtrain,Xtest['x'])
    #print(np.array(PreList))
    #print(np.array(Xtest['y']))

    accuracy = accuracy_score(np.array(Xtest['y']).astype(int), np.array(PreList).astype(int))
    precision = precision_score(np.array(Xtest['y']).astype(int), np.array(PreList).astype(int))
    recall = recall_score(np.array(Xtest['y']).astype(int), np.array(PreList).astype(int))
    f1 = f1_score(np.array(Xtest['y']).astype(int), np.array(PreList).astype(int))
    
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

if __name__ == '__main__':

    # 加载数据
    Xtrain = np.load(path + '/DDoS_Detection/CICDDoS2019/train.npz',allow_pickle=True)
    Xtest = np.load(path + '/DDoS_Detection/CICDDoS2019/test.npz',allow_pickle=True)

    # 归一化
    train_x = (Xtrain['x'] - np.mean(Xtrain['x'], axis=0)) / np.std(Xtrain['x'], axis=0)
    test_x = (Xtest['x'] - np.mean(Xtest['x'], axis=0)) / np.std(Xtest['x'], axis=0)

    test(Xtrain, Xtest)
