import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

class LogisticRegression(nn.Module):
    def __init__(self,n_feature):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_feature,1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def initialize_weights(m):
        if isinstance(m,nn.Linear):
            m.weight.data.normal_(0,0.01)
            m.bias.data.zero_()

    def forward(self,x):
        x = self.linear(x) 
        y = self.sigmoid(x)
        return y
    
    def Pre(self,X):
        y_pre = self.forward(X)
        y_pre[y_pre >= 0.5] = 1.0
        y_pre[y_pre < 0.5] = 0.0
        return y_pre
    
    def metric(self,Y,Y_pre):
        """
        Y: 标签值
        Y_pre: 预测值
        """
        accuracy = accuracy_score(Y,Y_pre)
        precision = precision_score(Y,Y_pre)    
        recall = recall_score(Y,Y_pre)
        f1 = f1_score(Y,Y_pre)
        return accuracy,precision,recall,f1

        