from re import L
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MGNN import *
# from MGNN import *
from model.GCN import *
# from GCN import *
from model.Utils import *
# from Utils import *
from torch.nn import Module


class Context(nn.Module):
    def __init__(self, DEVICE):
        super(Context, self).__init__()
        self.context = 5
        self.DEVICE = DEVICE

    def forward(self, x, y):
        assert self.context % 2 == 1, "context value error."
        cut = int(self.context / 2)
        tData = torch.zeros([x.shape[0] - 2 * cut, self.context, x.shape[1]]).to(self.DEVICE)
        label = torch.zeros([x.shape[0] - 2 * cut, y.shape[-1]]).to(self.DEVICE)
        for i in range(cut, x.shape[0] - cut):
            tData[i - cut] = x[i - cut:i + cut + 1]
            label[i - cut] = y[i]
        return tData, label


class DMSWGNN(nn.Module):
    def __init__(self, bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, temperature, scale, DEVICE, class_num):
        super(DMSWGNN, self).__init__()
        self.scale = scale
        self.AGCN = MGNN(bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, DEVICE)
        self.short_term_gru = nn.GRU(2 * out_dim * num_of_nodes, higru_hid, 1, True, True, 0.0, True)
        self.long_term_gru = nn.GRU(2 * higru_hid, higru_out, 1, True, True, 0.0, True)
        self.context = Context(DEVICE)

        #self.Linear2 = torch.nn.Linear(704, 64)
        self.Linear1 = torch.nn.Linear(2*higru_out, class_num)
        #self.tsg = tsGraph(DEVICE)

    def forward(self,d1,label):
        dd1 = self.AGCN(d1)

        short_input, label = self.context(dd1, label)
        # short_input = dd1
        # print('short_input:', short_input.shape)
        # print('label:', label.shape)
        out1, _ = self.short_term_gru(short_input)
        # print('out1:', out1.shape)
        long_input = out1[:, 2, :]  # 取中间时间片特征
        # long_input = out1
        long_input = long_input.unsqueeze(dim=0)  # reshape
        out, _ = self.long_term_gru(long_input)
        
        #out = self.tsg(torch.unsqueeze(self.Linear2(dd1),dim=0))
        result = self.Linear1(out.squeeze())
        # print('result:', result.shape)
        if len(result.shape) == 1:
            result = result.unsqueeze(0)

        model_loss = nn.CrossEntropyLoss()
        loss1 = model_loss(result, label)

        loss = loss1

        return result, loss, label


class classifier(nn.Module):
    def __init__(self, channel_num, class_num):
        super(classifier, self).__init__()
        self.channel_num = channel_num
        self.class_num = class_num
        self.mlp = nn.Sequential(
            nn.Linear(self.channel_num * 256, 256),
            nn.ReLU(),
            nn.Linear(256, class_num),
        )


    def forward(self, x, label):
        B, C, L = x.shape
        x = x.reshape(B, C*L)
        x = self.mlp(x)
        model_loss = nn.CrossEntropyLoss()
        loss = model_loss(x, label)

        return x, loss, x


def make_model(bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, temperature, scale, DEVICE, class_num):
    model = DMSWGNN(bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, temperature, scale, DEVICE, class_num)
    # model = classifier(num_of_nodes, class_num)
    model.float()

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model

