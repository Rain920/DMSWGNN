import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

class MR_DAKGNN(nn.Module):
    def __init__(self, bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, temperature, scale, DEVICE):
        super(MR_DAKGNN, self).__init__()
        self.scale = scale
        self.short_term_gru = nn.GRU(out_dim * num_of_nodes, higru_hid, 1, True, True, 0.0, True)
        self.long_term_gru = nn.GRU(2 * higru_hid, higru_out, 1, True, True, 0.0, True)
        self.Linear1 = torch.nn.Linear(2 * higru_out, 5)

    def forward(self, x,label):
        short_result, _ = self.short_term_gru(x)
        long_input = short_result[:, int(int(x.shape[1]) / 2), :]  # 取中间时间片特征
        long_input = long_input.unsqueeze(dim=0)  # reshape
        long_result, _ = self.long_term_gru(long_input)
        result = long_result.squeeze()
        result = self.Linear1(result)
        model_loss = nn.CrossEntropyLoss()
        loss1 = model_loss(result, label)
        loss = loss1
        return result,loss

def make_model(bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, temperature, scale, DEVICE):
    model = MR_DAKGNN(bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, temperature, scale, DEVICE)
    model.float()

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model