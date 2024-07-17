import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed
from collections import Counter
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from conf import parse
import os

args = parse.parse_args()

num = args.DecisionNum # 决策树的棵数
ratio = args.ratio # 子数据集样本数
k = args.featureNum # 随机选取的特征数
MaxDepth = args.MaxDepth # 最大深度
MinSample = args.MinSample # 最小样本数   

path = os.path.abspath(os.path.join(os.getcwd(), ".."))

# 加载数据
Xtrain = np.load(path + '/DDoS_Detection/CICDDoS2019/train.npz',allow_pickle=True)
Xtest = np.load(path + '/DDoS_Detection/CICDDoS2019/test.npz',allow_pickle=True)

# 归一化(Z-score归一化)
trainDataset = (Xtrain['x'] - np.mean(Xtrain['x'], axis=0)) / np.std(Xtrain['x'], axis=0)
testDataset = (Xtest['x'] - np.mean(Xtest['x'], axis=0)) / np.std(Xtest['x'], axis=0)
trainLabel = Xtrain['y']
testLabel = Xtest['y']

def gini(feature, index): # 计算基尼系数(评估合适阈值)
    """ 
    Dataset: 特征集
    index: 特征索引
    """
    data = feature.sort_values(by=index) # 对指定列进行排序
    values = data[index].unique() # 不同值
    ans = values[-1]
    bestGini = 1 # 最佳基尼系数
    for i in range(0,len(values) - 1):
        # 划分子数据集
        medium = (values[i] + values[i + 1]) / 2
        subDataset1 = data[data[index] <= medium] 
        subDataset2 = data[data[index] > medium] 
        # 计算比率（用于求总的基尼系数）
        Count1 = len(subDataset1) / len(data)
        Count2 = 1 - Count1
        # 求基尼系数
        PosPre = subDataset1['label'].sum() / float(len(subDataset1))
        NegPre = 1 - PosPre
        gini1 = 1 - pow(PosPre, 2) - pow(NegPre, 2) # 子数据集1基尼系数
        
        PosPre = subDataset2['label'].sum() / float(len(subDataset2))
        NegPre = 1 - PosPre
        gini2 = 1 - pow(PosPre, 2) - pow(NegPre, 2) # 子数据集1基尼系数
        # 总基尼系数
        Gini = Count1 * gini1 + Count2 * gini2
        
        if Gini < bestGini:
            bestGini = Gini
            ans = medium

    return bestGini,ans 

def BestFeature(Dataset): # 根据基尼系数选择最优特征
    """  
    feature: 特征集
    label: 标签
    """
    bestGini = 1
    res = 0
    flag = list(Dataset.columns)[0]
    for i in Dataset.columns: # 遍历特征
        if i != 'label':
            Gini, ans = gini(Dataset, i) # 计算基尼系数/阈值
        
            if Gini < bestGini:
                bestGini = Gini
                flag = i
                res = ans

    return flag,res      

def DecisionTree(Dataset, depth, ParentLabel, MaxDepth, MinSample): # 构建决策树（递归过程）
    """ 
    Dataset: 数据集
    """
    if len(Dataset) == 0:
        return ParentLabel
    if depth > MaxDepth: # 深度不可以太深
        return Dataset['label'].value_counts().idxmax()
    if len(Dataset) <= MinSample and len(Dataset) != 0: # 设置最小样本数
        return Dataset['label'].value_counts().idxmax()
    if Dataset['label'].sum() == len(Dataset) or Dataset['label'].sum() == 0: # 全部为同一类别
        return list(Dataset['label'])[0]
    
    flag, ans = BestFeature(Dataset) # 得到特征索引和阈值
    decisionDict = { flag : { ans:{} } } #决策树（字典形式）
    
    subDataset1 = Dataset[Dataset[flag] <= ans]
    subDataset2 = Dataset[Dataset[flag] > ans]
    
    decisionDict[flag][ans][0] = DecisionTree(subDataset1, depth + 1, (Dataset['label'].value_counts().idxmax()), MaxDepth, MinSample)
    decisionDict[flag][ans][1] = DecisionTree(subDataset2, depth + 1,(Dataset['label'].value_counts().idxmax()), MaxDepth, MinSample)

    return decisionDict   

def Predict(testData,decisionTree): # 预测结果
    """ 
    testData: 测试数据
    decisionTree: 决策树(字典)
    """
    if type(decisionTree).__name__ != 'dict':
        return decisionTree
    
    TreeDict = decisionTree[list(decisionTree.keys())[0]]
    TreeDict = TreeDict[list(TreeDict.keys())[0]]
    if testData[list(decisionTree.keys())[0]] < list(decisionTree[list(decisionTree.keys())[0]].keys())[0]:
        return Predict(testData,TreeDict[0])
    else:
        return Predict(testData,TreeDict[1])

def Metric(testDataset,testLabel, decisionTrees):
    """ 
    testDataset: 测试特征集
    testLabel: 测试标签
    decisionTree: 决策树集合
    """
    PredictionList = []
    for testData in testDataset:
        PreList = []
        for decisionTree in decisionTrees:
            PreList.append(Predict(testData,decisionTree))
        Prediction = Counter(PreList) 
        res, _ = Prediction.most_common(1)[0] # 少数服从多数
        PredictionList.append(int(res))
    
    accuracy = accuracy_score(np.array(testLabel).astype(int),np.array(PredictionList).astype(int))
    precision = precision_score(np.array(testLabel).astype(int),np.array(PredictionList).astype(int))
    recall = recall_score(np.array(testLabel).astype(int),np.array(PredictionList).astype(int))
    f1 = f1_score(np.array(testLabel).astype(int),np.array(PredictionList).astype(int))
    print('accuracy:',accuracy)
    print('precision:',precision)
    print('recall:',recall)
    print('f1:',f1)


def ParallelTree(Dataset,Label,ratio,k,random_state, MaxDepth, MinSample): # 并行建立多颗决策树
    """  
    Dataset: 原始数据集
    Label: 标签
    ratio: 子数据集样本比例
    k: 随机划分k种特征
    random_state: 随机种子
    """
    # 随机采样子数据集
    subDataset = Dataset.sample(n=int(ratio*len(Dataset)),replace=True,random_state=random_state).reset_index(drop=True)
    subFeature = random.sample(list(Dataset.columns),k) # 随机选取k种特征
    subDataset = subDataset.loc[:,subFeature] # 得到子数据集
    subLabel = Label.sample(n=int(ratio*len(Label)),replace=True,random_state=random_state).reset_index(drop=True)
      
    subDataset = subDataset.assign(label=subLabel.values.reshape(-1).tolist()) # 合并标签列
    
    decisionDict = DecisionTree(subDataset, 0, subDataset['label'].value_counts().idxmax(), MaxDepth, MinSample)
    return decisionDict

def train(num,Dataset, Label, ratio, k, MaxDepth, MinSample): # 训练过程(构建随机森林过程)
    """ 
    num: 决策树个数
    Dataset: 训练数据集
    Label: 标签值
    """
    random_states = None
    
    # 设定随机状态
    if random_states:
        random.seed(random_states)
    random_states = random.sample(range(num), num) # 得到列表

    # 并行处理
    # 得到一组决策树
    DecisionTrees = Parallel(n_jobs=-1, verbose=0, backend="multiprocessing")(
            delayed(ParallelTree)(Dataset,Label,ratio,k,random_state,MaxDepth,MinSample)
                for random_state in random_states)
    return DecisionTrees


if __name__ == '__main__':
    feature = pd.DataFrame(trainDataset)
    trainLabel = pd.DataFrame(trainLabel)

    DecisionTrees = train(num, feature, trainLabel, ratio, k, MaxDepth, MinSample)

    Metric(testDataset,testLabel,DecisionTrees)