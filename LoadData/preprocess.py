import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def LoadData(Path):
    """
    Path: 数据集文件路径
    """
    Data = np.load(Path,allow_pickle=True)
    return Data

def train_and_test(Data):
    """
    Data: 加载的数据
    """
    X = Data['x'] # 25种特征
    Y = Data['y'] # 标签
    
    class_0 = np.where(Y == 0)[0]
    class_1 = np.where(Y == 1)[0]
    
    class_count = 105042 # 按论文要求筛选数据

    # 随机抽取相同数量的每个类别的样本
    np.random.seed(2024) 
    balanced_class_0 = np.random.choice(class_0, class_count, replace=False)
    balanced_class_1 = np.random.choice(class_1, class_count, replace=False)

    # 打乱索引
    np.random.seed(2022) 
    np.random.shuffle(balanced_class_0) # 105042
    np.random.shuffle(balanced_class_1) # 105042
    
    train_balanced_class_0 = balanced_class_0[:int(0.8 * class_count)]
    test_balanced_class_0 = balanced_class_0[int(0.8 * class_count):]
    
    train_balanced_class_1 = balanced_class_1[:int(0.8 * class_count)]
    test_balanced_class_1 = balanced_class_1[int(0.8 * class_count):]
    
    # 合并两个类别的索引
    train_balanced = np.concatenate((train_balanced_class_0, train_balanced_class_1))
    test_balanced = np.concatenate((test_balanced_class_0, test_balanced_class_1))
    
    np.random.seed(2023) 
    np.random.shuffle(train_balanced)
    np.random.shuffle(test_balanced) 

    # 构建平衡的数据集
    # 划分数据集：训练集：测试集 = 8 : 2
    X_train = X[train_balanced]
    Y_train = Y[train_balanced]
    
    X_test = X[test_balanced]
    Y_test = Y[test_balanced]
    
    np.savez(
        "../CICDDoS2019/train",
        x=X_train,
        y=Y_train,
    )
    np.savez(
        "../CICDDoS2019/test",
        x=X_test,
        y=Y_test,
    )
    
    return X_train, X_test, Y_train, Y_test
    
def ShowCount(Data):
    """
    Data: 待查看的数据
    """
    Data = pd.DataFrame(Data)
    Count = Data.value_counts()
    print(Count)

if __name__ == "__main__":
    Data = LoadData('../CICDDoS2019/Data.npz') # 加载数据
    X_train, X_test, Y_train, Y_test = train_and_test(Data) # 划分数据集
    ShowCount(Y_test) # 显示标签数量  合法流量：112612  攻击流量：107318
