import torch
import torch.nn as nn
import torch.nn.functional as F


class graph_construction(nn.Module):
    def __init__(self, bandwidth, num_of_nodes, DEVICE):
        super(graph_construction, self).__init__()
        self.bandwidth = bandwidth
        self.DEVICE = DEVICE
        self.weight = nn.Parameter(torch.rand(num_of_nodes, num_of_nodes).to(DEVICE),requires_grad=True)

    def forward(self, x):
        N, V, C = x.shape
        S = x.unsqueeze(dim=0)
        S = torch.repeat_interleave(S, V, dim=0)
        S1 = torch.transpose(S, 0, 1)
        S2 = torch.transpose(S1, 1, 2)
        diff = S1 - S2  # [batch,num_of_nodes,num_of_nodes,num_of_features]
        L2fanshu = torch.norm(diff, 2, dim=-1)  # [batch,num_of_nodes,num_of_nodes]
        adj = torch.exp(-0.5 / (self.bandwidth ** 2) * L2fanshu)
        adj = torch.matmul(adj,self.weight)
        return F.softmax(adj,dim=-1)

class graph_construction1(nn.Module):
    def __init__(self, bandwidth, DEVICE):
        super(graph_construction1, self).__init__()
        self.bandwidth = bandwidth
        self.DEVICE = DEVICE

    def forward(self, x):
        N, V, C = x.shape
        S = x.unsqueeze(dim=0)
        S = torch.repeat_interleave(S, V, dim=0)
        S1 = torch.transpose(S, 0, 1)
        S2 = torch.transpose(S1, 1, 2)
        diff = S1 - S2  # [batch,num_of_nodes,num_of_nodes,num_of_features]
        L2fanshu = torch.norm(diff, 2, dim=-1)  # [batch,num_of_nodes,num_of_nodes]
        adj = torch.exp(-0.5 / (self.bandwidth ** 2) * L2fanshu)
        return F.softmax(adj,dim=-1)
