import os
import numpy as np
import torch
from model.Utils import *

class kFoldGenerator():
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

    # Initializate
    def __init__(self, k, d1, y):
        if len(d1)!=k or len(y)!=k:
            assert False,'Data generator: Length of x or y is not equal to k.'
        self.k=k
        self.d1_list = d1
        self.y_list=y

    # Get i-th fold
    def getFold(self, i):
        isFirst=True
        train_len = []
        train_lab = []
        for p in range(self.k):
            if p!=i:
                if isFirst:
                    train_data_d1 = self.d1_list[p]
                    train_targets = AddContext(self.y_list[p],5, label=True)
                    #train_targets = self.y_list[p]

                    len = np.array(train_data_d1).shape[0]
                    train_len.append(len)
                    lllen = np.array(train_targets).shape[0]
                    train_lab.append(lllen)
                    isFirst = False
                else:
                    d1 = self.d1_list[p]
                    train_data_d1 = np.concatenate((train_data_d1, d1))
                    lab = AddContext(self.y_list[p],5, label=True)
                    #lab = self.y_list[p]
                    train_targets   = np.concatenate((train_targets, lab))
                    len = np.array(d1).shape[0]
                    train_len.append(len)
                    lllen = np.array(lab).shape[0]
                    train_lab.append(lllen)
            else:
                val_data_d1 = self.d1_list[p]
                val_targets = AddContext(self.y_list[p],5, label=True)
                #val_targets = self.y_list[p]


        num_val = np.array(val_data_d1).shape[0]
        val_lab = np.array(val_targets).shape[0]

        train_data_d1 = torch.from_numpy(train_data_d1.astype(np.float32))
        train_data_d1 = torch.FloatTensor(train_data_d1)
        train_targets = torch.LongTensor(train_targets)
        train_targets = train_targets.squeeze()
        train_len = np.array(train_len)

        val_data_d1 = torch.from_numpy(val_data_d1.astype(np.float32))
        val_data_d1 = torch.FloatTensor(val_data_d1)
        val_targets = torch.LongTensor(val_targets)
        val_targets = val_targets.squeeze()


        return train_targets,val_targets, train_len,train_lab,num_val,val_lab

    # Get all data x
    def getX(self):
        All_X = self.x_list[0]
        for i in range(1,self.k):
            All_X = np.append(All_X,self.x_list[i], axis=0)
        return All_X

    # Get all label y
    def getY(self):
        All_Y = self.y_list[0]
        for i in range(1,self.k):
            All_Y = np.append(All_Y,self.y_list[i], axis=0)
        return All_Y

    # Get all label y
    def getY_one_hot(self):
        All_Y = self.getY()
        return np.argmax(All_Y, axis=1)

