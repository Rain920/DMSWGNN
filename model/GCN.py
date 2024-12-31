import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

class GCN(nn.Module):
    def __init__(self, K, in_dim, out_dim, num_of_nodes, DEVICE):
        super(GCN, self).__init__()
        self.layer1 = GCN_Layer(in_dim=in_dim,out_dim=128,num_of_nodes=num_of_nodes,DEVICE=DEVICE)
        self.layer2 = GCN_Layer(in_dim=128, out_dim=out_dim, num_of_nodes=num_of_nodes, DEVICE=DEVICE)
        self.liner = torch.nn.Linear(2000, 32)

    def forward(self, x, adj):
        #进行K层自适应图卷积
        hid = self.layer1(x, adj)
        hid = self.layer2(hid,adj)

        #使用残差连接防止过平滑
        #res = F.relu(self.liner(x))
        output = torch.sigmoid(hid)
        return output

class GCN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, num_of_nodes, DEVICE):
        super(GCN_Layer, self).__init__()
        self.DEVICE = DEVICE
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_of_nodes = num_of_nodes
        self.theta = torch.nn.Parameter(torch.rand(in_dim, out_dim).to(DEVICE),requires_grad=True)

    def forward(self, x, adj):
        batch_size, num_of_nodes, in_channels = x.shape
        x = torch.matmul(x,self.theta)
        h = torch.matmul(adj, x)
        gcn = F.relu(h)
        return gcn