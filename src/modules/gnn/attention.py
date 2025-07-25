import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, input_dim, nhid, heads, dropout):
        super().__init__()
        p_feats_num = input_dim
        d_feats_num = input_dim
        self.heads = heads
        self.linear_p = nn.Linear(p_feats_num, nhid)
        self.q_p = nn.Parameter(torch.rand(nhid, self.heads))
        self.linear_d = nn.Linear(d_feats_num, nhid)
        self.q_d = nn.Parameter(torch.rand(nhid, self.heads))
        self.att_fusion = nn.Linear(self.heads * nhid, nhid)

    

    def forward(self, c, l):
        p_feats, d_feats = c, l
        
        p_feats = torch.tanh(self.linear_p(p_feats))
        p_alpha = torch.matmul(p_feats, self.q_p)

        d_feats = torch.tanh(self.linear_d(d_feats))
        d_alpha = torch.matmul(d_feats, self.q_d)

        alpha = torch.exp(p_alpha) + torch.exp(d_alpha)
        p_alpha = torch.exp(p_alpha) / alpha
        d_alpha = torch.exp(d_alpha) / alpha
        fusion_x = torch.cat(
            [p_alpha[:, i].view(-1, 1) * p_feats + d_alpha[:, i].view(-1, 1) * d_feats for i in range(self.heads)],
            dim=1)
        fusion_x = self.att_fusion(fusion_x)
        return fusion_x 