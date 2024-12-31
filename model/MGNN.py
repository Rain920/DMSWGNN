import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from model.Graph_Construction import *
# from Graph_Construction import *
from model.GCN import *
# from GCN import *
from model.kernel import *
# from kernel import *
from torch.nn import Module



class MGNN(nn.Module):
    def __init__(self, bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, DEVICE):
        super(MGNN, self).__init__()
        self.DEVICE = DEVICE
        self.graph_construction = graph_construction(bandwidth, num_of_nodes, DEVICE)
        self.degree = 3
        self.AKGCN = GCN(K, in_dim, out_dim, num_of_nodes, DEVICE)
        self.linear = torch.nn.Linear(256, 128)
        #self.ww = torch.nn.Parameter(torch.rand(2, 1).to(DEVICE), requires_grad=True)

    def forward(self, x):
        # 图构建，用中间时间片构建图
        graph_x = x
        # print('graph_x:', graph_x.shape)
        adj = self.graph_construction(graph_x)
        # print('adj:', adj)
        b_kernel = cal_b_spline(self.degree, adj)
        wavelet_kernel = cal_spline_wavelet(self.degree, adj)
        #自适应核图卷积
        gcn1 = self.AKGCN(x, b_kernel)
        # print('gcn1:', gcn1.shape)
        gcn2 = self.AKGCN(x, wavelet_kernel)
        # print('gcn2:', gcn2.shape)
        #akgcn = F.relu(gcn1+gcn2)
        #w = nn.Softmax(dim=0)(self.ww)
        akgcn = torch.cat((gcn1,gcn2),dim=-1)
        # print('akgcn:', akgcn.shape)
        #akgcn = gcn1 + gcn2
        akgcn = F.relu(akgcn + self.linear(x))

        akgcn = akgcn.flatten(start_dim=-2, end_dim=-1)
        return akgcn

