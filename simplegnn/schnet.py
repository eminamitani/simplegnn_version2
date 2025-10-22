import torch
import torch.nn as nn
import torch.nn.functional as F
from simplegnn.util import make_radial, make_envelope
from math import pi as PI
import numpy as np


# ==============================================================
# 原子種を埋め込みベクトルに変換
# ==============================================================
class TypeEmbedding(nn.Module):
    def __init__(self, type_num, type_dim):
        super().__init__()
        self.embedding = nn.Embedding(type_num, type_dim)

    def forward(self, x):
        return self.embedding(x)
    
class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.softplus(x) - torch.log(torch.tensor(2.0, device=x.device))


# ==============================================================
# 相互作用ブロック (Interaction Block)
class InteractionBlock(nn.Module):
    def __init__(self, hidden_dim, n_radial, num_filters, cutoff,
                 envelope_type:str='smoothstep'):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_radial, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.cutoff = cutoff
        self.envelope = make_envelope(envelope_type, cutoff)
        self.lin1 = nn.Linear(hidden_dim, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, hidden_dim)
        self.act = ShiftedSoftplus()

    def forward(self, x, edge_index, edge_weight, edge_attr):

        # 原子間距離の計算
        distances = torch.norm(edge_weight, dim=-1)  # 原子間距離 (num_edges,)
        envelope = self.envelope(distances)  # カットオフ関数 (num_edges,)
        
        # フィルター重みの計算
        W = self.mlp(edge_attr) * envelope  # (num_edges, num_filters)

        # メッセージ生成
        i, j = edge_index  # edge_index (2, num_edges)
        messages = W * self.lin1(x[j])  # (num_edges, num_filters)

        #print("messages:",messages.shape)
        #print("x:",x.shape)

        # メッセージ集約
        #index_addの利用を避ける（indexが同じものがあるケースだと問題がある）
        agg_messages = torch.zeros_like(self.lin1(x))
        # scatter_addが入った状態でgradが通るように、行列形状を調整している
        index = i.unsqueeze(1) if i.ndim == 1 else index
        agg_messages = torch.scatter_add(agg_messages, 0, index.expand_as(messages), messages)

        #print("agg_messages:",agg_messages.shape)
        # 特徴量更新
        h = self.act(self.lin2(agg_messages))  # 非線形変換を適用
        #print("h:",h.shape)
        return x + h


class SchNet_dict():
    def __init__(self, hidden_dim, n_radial, num_filters, num_interactions, cutoff, type_num=100):
        self.hidden_dim = hidden_dim
        self.n_radial = n_radial
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.type_num = type_num

    def to_dict(self):
        return {
            "hidden_dim": self.hidden_dim,
            "n_radial": self.n_radial,
            "num_filters": self.num_filters,
            "num_interactions": self.num_interactions,
            "cutoff": self.cutoff,
            "type_num": self.type_num
        }

    @classmethod
    def from_dict(cls, dic):
        return cls(**dic)
    
class SchNetModel(nn.Module):


    def __init__(self, hidden_dim, n_radial, num_filters, 
                 num_interactions, cutoff, type_num=100,
                 radial_type='gaussian',
                 radial_kwargs={}):
        super().__init__()
        self.cutoff=cutoff
        self.setups=SchNet_dict(hidden_dim, n_radial, num_filters, num_interactions, cutoff, type_num)
        self.embedding = TypeEmbedding(type_num, hidden_dim)
        self.radial= make_radial(radial_type, n_radial, cutoff, **radial_kwargs)

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_dim, n_radial,
                                     num_filters, cutoff)
            self.interactions.append(block)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim // 2, 1)
        )
        #initialize weights
        #pytorch のデフォルト初期化で十分であった（下手にいじると収束しにくくなる）
        #self.apply(initialize_weights)

    def forward(self, x, edge_index, edge_weight, batch=None):
        #力を計算したいので、edge_weight (Rj-Ri)の微分を取る
        edge_weight.requires_grad_()
        # 埋め込み
        h = self.embedding(x)

        #print("embedding:",h.shape)

        # basis functionの計算
        distances = torch.norm(edge_weight, dim=-1)
        rbf_expansion = self.radial(distances)
        #print("rbf:",rbf_expansion.shape)

        # 相互作用ブロックを適用
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_weight, rbf_expansion)

        # 出力層
        energy = self.output(h)

        # derivative with respect to edge_weight=rj-ri
        diff_E = torch.autograd.grad(energy.sum(), edge_weight, create_graph=True)[0]
        #p: pair, k,l: x,y,z
        sigma_ij = torch.einsum('pk,pl->pkl', edge_weight, diff_E)


        #edge_weightはrj-riである。
        #ここでdeviceに指定していないと、CPUで計算されてしまう
        force_i = torch.zeros((len(x), 3), device=edge_weight.device)
        force_j = torch.zeros((len(x), 3), device=edge_weight.device)

        #scatter_add version (index_addだとindexが同じものがあるケースだと問題があるかも)
        index_i=edge_index[0].unsqueeze(1) if edge_index[0].ndim == 1 else edge_index[0]
        index_j=edge_index[1].unsqueeze(1) if edge_index[1].ndim == 1 else edge_index[1]
        force_i=torch.scatter_add(force_i, 0, index_i.expand_as(diff_E), diff_E)
        force_j=torch.scatter_add(force_j, 0, index_j.expand_as(diff_E), -diff_E)

        forces=force_i+force_j

        # バッチごとに集約


        if batch is not None:
            batch_max = batch.max().item()
            total_energy = torch.zeros(batch_max + 1, device=energy.device)
            total_energy = total_energy.index_add_(0, batch, energy.squeeze())

            sigma = torch.zeros((batch_max + 1, 3, 3), device=edge_weight.device)
            batch_edge = batch[edge_index[0]]  # edge_indexのi側の原子のバッチ情報
            sigma = sigma.index_add_(0, batch_edge, sigma_ij)

        
        else:
            total_energy = energy.sum()
            sigma = sigma_ij.sum(dim=0)
            
        if sigma.dim() == 3:
            sigma = 0.5 * (sigma + sigma.transpose(1, 2))
        else:
            sigma = 0.5 * (sigma + sigma.T)

        return total_energy,forces,sigma




