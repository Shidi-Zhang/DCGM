from .SetTransformer import SAB
from .gnn.GNNs import GNNGraph
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from .gnn.kan import KANLinear
from .gnn.geognn import GINGraphPooling

class KAN(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(KAN, self).__init__()

        self.h1 = KANLinear(embedding_size, 32)

        self.gate_layer = nn.Linear(32, 1)

    def forward(self, seqs):
        gates = self.gate_layer(self.h1(seqs))
        output = F.sigmoid(gates)

        return output
class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(self, main_feat, other_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)

        output = torch.matmul(Attn, other_feat)  # (N_coarse, dim)
        return output



class DCGM(torch.nn.Module):
    def __init__(
        self, global_para, substruct_para, emb_dim, voc_size,
        substruct_num, global_dim, substruct_dim, use_embedding=False,
        device=torch.device('cuda:0'), dropout=0.5, *args, **kwargs
    ):
        super(DCGM, self).__init__(*args, **kwargs)
        self.device = device
        self.use_embedding = use_embedding
        self.poly = KAN(emb_dim * 2)
        self.fai = 0.05
        self.emb_dim=emb_dim
        if self.use_embedding:
            self.substruct_emb = torch.nn.Parameter(
                torch.zeros(substruct_num, emb_dim)
            )
        else:
            self.substruct_encoder = GNNGraph(**substruct_para)

        self.global_encoder = GINGraphPooling(**global_para)

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim),
            torch.nn.Embedding(voc_size[2], emb_dim)
        ])
        self.encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()
        self.sab = SAB(substruct_dim, substruct_dim, 2, use_ln=True)
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 6, emb_dim)
        )
        self.substruct_rela = torch.nn.Linear(emb_dim, substruct_num)
        self.global_rela = torch.nn.Linear(emb_dim, voc_size[2])
        self.drug_query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, voc_size[2])
        )
        self.aggregator = AdjAttenAgger(
            global_dim, substruct_dim, max(global_dim, substruct_dim)
        )
        score_extractor = [
            torch.nn.Linear(2*substruct_dim, substruct_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(substruct_dim // 2, 1)
        ]
        self.score_extractor = torch.nn.Sequential(*score_extractor)
        self.linear_layer = nn.Linear(in_features=64, out_features=128)
        self.seq_gru = torch.nn.GRU(voc_size[2], voc_size[2], batch_first=True)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        if self.use_embedding:
            torch.nn.init.xavier_uniform_(self.substruct_emb)

    def forward(
        self, substruct_data, mol_data, patient_data,
        ddi_mask_H, tensor_ddi_adj, average_projection
    ):

        diag_seq = []
        proc_seq = []
        med_seq = []
        preser = []
        # def sum_embedding(embedding):
        #     return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        if len(patient_data) == 1:
            # 只有当前这次就诊，取诊断和手术信息构造 query，不使用药物信息
            cur_adm = patient_data[0]
            Idx1 = torch.LongTensor([cur_adm[0]]).to(self.device)
            Idx2 = torch.LongTensor([cur_adm[1]]).to(self.device)

            diag_seq = torch.sum(self.rnn_dropout(self.embeddings[0](Idx1)), dim=1, keepdim=True)
            proc_seq = torch.sum(self.rnn_dropout(self.embeddings[1](Idx2)), dim=1, keepdim=True)
            med_seq = torch.zeros((1, 1, self.emb_dim)).to(self.device)

        else:
            for adm in patient_data:
                Idx1 = torch.LongTensor([adm[0]]).to(self.device)
                Idx2 = torch.LongTensor([adm[1]]).to(self.device)
                repr1 = torch.sum(self.rnn_dropout(self.embeddings[0](Idx1)), dim=1, keepdim=True)
                repr2 = torch.sum(self.rnn_dropout(self.embeddings[1](Idx2)), dim=1, keepdim=True)
                diag_seq.append(repr1)
                proc_seq.append(repr2)
                Idx3 = torch.LongTensor([adm[2]]).to(self.device)
                repr3 = torch.sum(self.rnn_dropout(self.embeddings[2](Idx3)), dim=1, keepdim=True)
                med_seq.append(repr3)
            diag_seq = torch.cat(diag_seq, dim=1)
            proc_seq = torch.cat(proc_seq, dim=1)
            med_seq = torch.cat(med_seq, dim=1)
            # 表明只取前t-1次的med
            # if med_seq.size(1) > 1:
            #     med_seq = med_seq[:, :-1, :]
            # else:
            #     med_seq = torch.zeros_like(med_seq[:, :1, :])
            patient_representation = torch.cat([diag_seq, proc_seq], dim=-1).squeeze(dim=0)

            cur_query = patient_representation[-1:, :]

            # 获取与当前患者相似的第i次诊断
            poly_cur = self.poly(cur_query)
            for i in range(len(patient_data) - 1):

                poly_his = self.poly(patient_representation[i])
                s = abs(poly_cur - poly_his)
                if s <= self.fai:
                    preser.append(i)

            if not preser:
                diag_seq = diag_seq[:, -1:, :]
                proc_seq = proc_seq[:, -1:, :]
                med_seq = torch.zeros((1, 1, self.emb_dim), device=self.device)

            else:
                # residuals_drug_emb = torch.cat([residuals_drug_emb[i] for i in preser],dim=0)
                preser.append(len(patient_data) - 1)
                diag_seq = torch.cat([diag_seq[:, i:i + 1, :] for i in preser], dim=1)
                proc_seq = torch.cat([proc_seq[:, i:i + 1, :] for i in preser], dim=1)
                # 排除最后一个时间步
                filtered_preser = [i for i in preser if i < med_seq.shape[1] - 1]
                if filtered_preser:
                    med_seq = torch.cat([med_seq[:, i:i + 1, :] for i in filtered_preser], dim=1)
                else:
                    # 若无历史，给默认全零张量
                    med_seq = torch.zeros((1, 1, self.emb_dim), device=self.device)

        output1, hidden1 = self.encoders[0](diag_seq)
        output2, hidden2 = self.encoders[1](proc_seq)
        output3, hidden3 = self.encoders[2](med_seq)
        seq_repr = torch.cat([hidden1, hidden2,hidden3], dim=-1)
        last_repr = torch.cat([output1[:, -1], output2[:, -1],output3[:, -1]], dim=-1)
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])

        query = self.query(patient_repr)


        substruct_weight = torch.sigmoid(self.substruct_rela(query))
        global_weight = torch.sigmoid(self.global_rela(query))



        global_embeddings = self.global_encoder(**mol_data)
        global_embeddings = torch.mm(average_projection, global_embeddings)


        # 计算粗粒度的注意力
        # coarse_attention_weights = torch.softmax(torch.matmul(global_weight,global_embeddings),dim=-1)  # [1] -> attention score
        # coarse_weighted_embedding = global_weight * global_embeddings #[131,64]
        coarse_weighted_embedding = global_weight.unsqueeze(-1) * global_embeddings


        substruct_embeddings = self.sab(
            self.substruct_emb.unsqueeze(0) if self.use_embedding else
            self.substruct_encoder(**substruct_data).unsqueeze(0)
        ).squeeze(0)

        #计算细粒度的注意力
        # 加权
        fine_weighted_embedding = substruct_weight.unsqueeze(-1) * substruct_embeddings

        fused_repr = self.aggregator(
            coarse_weighted_embedding, fine_weighted_embedding, mask=torch.logical_not(ddi_mask_H > 0)
        )

        # 拼接 coarse + 融合后的细粒度
        molecule_embeddings = torch.cat([coarse_weighted_embedding, fused_repr], dim=-1)  # (N_coarse, dim*2)



        score = self.score_extractor(molecule_embeddings).t()

        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum()
        return score, batch_neg

