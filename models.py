import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sparse_dropout, dot_product_scipy, sparse_transpose
import numpy as np
import math

class Top_HiCL_Matching(nn.Module):
    def __init__(self, E_j, E_s, device, metric="avg", freeze=False):
        super(Top_HiCL_Matching, self).__init__()
        self.E_j = nn.Embedding.from_pretrained(E_j, freeze=freeze)
        self.E_s = nn.Embedding.from_pretrained(E_s, freeze=freeze)

        if metric== "avg":
            self.pool = MeanPooling()
        elif metric == "att":
            self.pool = AttentionPooling(E_s.size(1), device)
        elif metric == "matt":
            self.pool = MultiHeadAttentionPooling(E_s.size(1), device=device)
        elif metric =="sum":
            self.pool = SumPooling()

    def forward(self, user_jobs, user_skills, test=False, k=5, ):
        # skills = self.E_s[user_skills]
        if test==True:
            labels = list()
            topk_preds, acc_preds, ncdg_mrr_preds, all_preds = list(), list(), list(), list()
            topk_preds5,topk_preds10 = list(), list()
            for jid, sids in zip(user_jobs, user_skills):
                try:
                    if len(sids)==0: continue
                    user_skill_emb = self.pool(self.E_s(sids))
                    # Top-K
                    scores = self.E_j.weight @ user_skill_emb.T  # Innder Product
                    scores = F.softmax(scores / 0.1, dim=0)
                    topk_values_3, topk_indices_3 = torch.topk(scores, k=3)
                    topk_values_5, topk_indices_5 = torch.topk(scores, k=5)
                    topk_values_10, topk_indices_10 = torch.topk(scores, k=10)
                    topk_preds.append(topk_indices_3.tolist())
                    topk_preds5.append(topk_indices_5.tolist())
                    topk_preds10.append(topk_indices_10.tolist())
                    # Accuracy
                    acc_preds.append(topk_indices_10[0])
                    # ndcg, mrr
                    ncdg_mrr_preds.append(topk_indices_10)
                    # all
                    all_preds.append(scores)
                    # label
                    labels.append(jid)
                except:
                    pass
            
            return (labels, topk_preds, topk_preds5, topk_preds10, acc_preds, ncdg_mrr_preds, all_preds)
        else:
            eps = 1e-9
            loss = 0
            nums = 0
            for jid, sids in zip(user_jobs, user_skills):
                if len(sids)==0: continue
                user_skill_emb = self.pool(self.E_s(sids))
                
                scores = self.E_j.weight @ user_skill_emb.T
                probs = F.softmax(scores / 0.1, dim=0)
                onehot = torch.zeros_like(probs)
                onehot[jid] = 1.0
                loss_batch = -(onehot * torch.log(probs + eps)).sum()
                loss += loss_batch
                nums += 1
            return loss / nums
    

class Top_HiCL(nn.Module):
    def __init__(self, e_j_f, e_s_f,  num_layers, adj_j_norm, adj_s_norm, dim, temp, lambda_1, activation, dropout, bias, device, layer_name="GCN"):
        """
        Parameter
        j_f: Job Feature(Text ==> Embeddings by BERT, dim: 768)
        s_f: Skill Feature(Text ==> Embeddings by BERT, dim: 768)
        """
        super(Top_HiCL, self).__init__()
        self.E_j_0 = e_j_f # job init Embeddings
        self.E_s_0 = e_s_f # skill init Embeddings


        self.adj_j_norm = adj_j_norm
        self.adj_s_norm = adj_s_norm
        self.layer_name = layer_name

        # original
        self.E_j_list = [None] * (num_layers + 1)
        self.E_s_list = [None] * (num_layers + 1)
        self.E_j_list[0] = self.E_j_0
        self.E_s_list[0] = self.E_s_0

        # Augmented
        self.G_j_list = [None] * (num_layers + 1)
        self.G_s_list = [None] * (num_layers + 1)
        self.G_j_list[0] = None
        self.G_s_list[0] = None

        self.temp = temp # Temperature
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation # nn.LeakyReLU(1e-2)
        self.device = device
        self.dim = dim
        self.lambda_1 = lambda_1
        self.E_j = None
        self.E_s = None
        self.G_j = None
        self.G_s = None

        # Learnable Params
        self.W_j = nn.ParameterList([nn.Linear(self.dim, self.dim, bias=bias) for idx in range(num_layers)])
        self.W_s = nn.ParameterList([nn.Linear(self.dim, self.dim, bias=bias) for idx in range(num_layers)])
        self.W_s_aug = nn.ParameterList([nn.Linear(self.dim, self.dim, bias=bias) for idx in range(num_layers)])
        self.W_j_aug = nn.ParameterList([nn.Linear(self.dim, self.dim, bias=bias) for idx in range(num_layers)])

        # Attention
        if self.layer_name == "GAT":
            self.heads = 4
            self.att_Q = nn.ModuleList([nn.Linear(self.dim,
                                                self.dim, bias=False)
                                        for l in range(self.num_layers)])
            self.att_K = nn.ModuleList([nn.Linear(self.dim,
                                                self.dim, bias=False)
                                        for l in range(self.num_layers)])

            self.att_Q_j = nn.ModuleList([
                nn.Linear(self.dim, self.dim, bias=False)
                for l in range(self.num_layers)])
            self.att_K_s = nn.ModuleList([
                nn.Linear(self.dim, self.dim, bias=False)
                for l in range(self.num_layers)])

    def update_aug_embeddings(self,aug_e_j, aug_e_s, device):
        self.aug_e_j = F.normalize(aug_e_j, p=2, dim=1).to(device) # augmented job embeddings
        self.aug_e_s = F.normalize(aug_e_s, p=2, dim=1).to(device) # augmented skill embeddings
        if self.G_j_list[0] == None:
            self.G_j_list[0] = self.aug_e_j.to(device)
            self.G_s_list[0] = self.aug_e_s.to(device)


    def forward(self,j_ids,s_ids, negs):

        for layer in range(1, self.num_layers+1):
            # GCN
            if isinstance(self.E_j_list[layer-1], list):
                self.E_j_list[layer-1] = torch.stack(self.E_j_list[layer-1]).to(self.device).squeeze(1)
                self.E_s_list[layer-1] = torch.stack(self.E_s_list[layer-1]).to(self.device).squeeze(1)
            
            if self.layer_name == "GCN":
                self.E_j_list[layer] = self.E_j_list[layer-1] + self.activation(self.W_j[layer-1](dot_product_scipy(sparse_dropout(self.adj_j_norm,self.dropout, self.device), self.E_s_list[layer-1])))
                self.E_s_list[layer] = self.E_s_list[layer-1] + self.activation(self.W_s[layer-1](dot_product_scipy(sparse_transpose(sparse_dropout(self.adj_s_norm,self.dropout, self.device)), self.E_j_list[layer-1])))
                # # Augmented GCN
                self.G_j_list[layer] =  self.G_j_list[layer-1].detach() + self.activation(self.W_j_aug[layer-1](dot_product_scipy(sparse_dropout(self.adj_j_norm,self.dropout, self.device), self.G_s_list[layer-1].detach())))
                self.G_s_list[layer] =  self.G_s_list[layer-1].detach() + self.activation(self.W_s_aug[layer-1](dot_product_scipy(sparse_transpose(sparse_dropout(self.adj_s_norm,self.dropout, self.device)), self.G_j_list[layer-1].detach())))
            elif self.layer_name == "GAT":
                msg_js = sparse_attention_agg(
                    sparse_dropout(self.adj_j_norm, self.dropout, self.device),
                    self.E_j_list[layer-1], self.E_s_list[layer-1],
                    self.att_Q[layer-1], self.att_K[layer-1],
                    self.leaky_relu, self.dropout)
                self.E_j_list[layer] = self.E_j_list[layer-1] + \
                    self.activation(self.W_j[layer-1](msg_js))

                # --- SKILL <= JOB  (attention) ---
                msg_sj = sparse_attention_agg(
                    sparse_transpose(sparse_dropout(
                        self.adj_s_norm, self.dropout, self.device)),
                    self.E_s_list[layer-1], self.E_j_list[layer-1],
                    self.att_Q[layer-1], self.att_K[layer-1],
                    self.leaky_relu, self.dropout)
                self.E_s_list[layer] = self.E_s_list[layer-1] + \
                    self.activation(self.W_s[layer-1](msg_sj))

                # --- Augmented ---
                self.G_j_list[layer] = self.G_j_list[layer-1].detach() + \
                    self.activation(self.W_j_aug[layer-1](
                        sparse_attention_agg(
                            sparse_dropout(self.adj_j_norm, self.dropout, self.device),
                            self.G_j_list[layer-1].detach(),
                            self.G_s_list[layer-1].detach(),
                            self.att_Q[layer-1], self.att_K[layer-1],
                            self.leaky_relu, self.dropout)))

                self.G_s_list[layer] = self.G_s_list[layer-1].detach() + \
                    self.activation(self.W_s_aug[layer-1](
                        sparse_attention_agg(
                            sparse_transpose(sparse_dropout(
                                self.adj_s_norm, self.dropout, self.device)),
                            self.G_s_list[layer-1].detach(),
                            self.G_j_list[layer-1].detach(),
                            self.att_Q[layer-1], self.att_K[layer-1],
                            self.leaky_relu, self.dropout)))

        # aggregate & norm
        self.E_j = F.normalize(sum(self.E_j_list) / len(self.E_j_list), p=2, dim=1)
        self.E_s = F.normalize(sum(self.E_s_list) / len(self.E_s_list), p=2, dim=1)
        self.G_j = F.normalize(sum(self.G_j_list) / len(self.G_j_list), p=2, dim=1).detach()
        self.G_s = F.normalize(sum(self.G_s_list) / len(self.G_s_list), p=2, dim=1).detach()

        # InfoNCE Loss
        # jids, sids: positive
        negs = torch.stack(negs)
        neg_emb = self.E_s[negs] # (n_neg, batch_size, emb_dim)
        neg_emb_T = neg_emb.permute(1, 0, 2).transpose(1,2)
        neg_score = torch.log(torch.exp(self.G_j[j_ids] @ self.E_j[j_ids].T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(self.G_s[s_ids] @ neg_emb_T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((self.G_j[j_ids] * self.E_j[j_ids]).sum(1) / self.temp,-1.0,1.0)).mean() + (torch.clamp((self.G_s[s_ids] * self.E_s[s_ids]).sum(1) / self.temp,-1.0,1.0)).mean()
        loss_cl = (-pos_score + neg_score) * 0.2

        # reg loss
        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()
        loss_reg = loss_reg * self.lambda_1

        loss = loss_cl + loss_reg
        loss = torch.tensor(loss, dtype=torch.float32, device=self.device) if isinstance(loss, float) else loss
        loss_cl = torch.tensor(loss_cl, dtype=torch.float32, device=self.device) if isinstance(loss_cl, float) else loss_cl
        loss_reg = torch.tensor(loss_reg, dtype=torch.float32, device=self.device) if isinstance(loss_reg, float) else loss_reg

        return (loss, loss_cl, loss_reg)
    

class Top_HiCL_H(nn.Module):
    def __init__(self, emb_s, pos_dim, dim, depth, num_layers, lambda_1, activation, temp, dropout, bias, device, layers_name="GCN", heads=1):
        super(Top_HiCL_H, self).__init__()

        self.emb_s = emb_s
        self.depth = depth
        self.emb_p = nn.Embedding(depth, pos_dim)
        self.embeddings_list = [None] * (num_layers + 1)
        self.embeddings_list[0]= self.emb_s.detach().cpu()
        self.embeddings = None
        self.layers_name = layers_name
        # Projection
        self.projection_layer = nn.Linear(dim+pos_dim, dim)

        # GCN
        self.layers = nn.ModuleList()
        if layers_name == "GCN":
            self.layers.append(nn.Linear(dim, dim, bias=bias))
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(dim, dim, bias=bias))
        elif layers_name == "GAT":
            for l in range(num_layers):
                self.layers.append(
                    GraphAttentionLayer(
                        in_dim=dim,
                        out_dim=dim,
                        heads=heads,
                        dropout=dropout,
                        bias=bias)
                )
        self.output_layer = nn.Linear(dim, dim, bias=bias)
        self.lambda_1 = lambda_1
        self.activation = activation
        # Dropout
        self.temp = temp
        self.dropout = dropout
        self.device = device

    def forward(self, adj_norm, sids, position_ids, pos, negs):
        """
        Forward pass for hierarchical GCN.
        :param adj_norm: normalized full adjacency matrix (N x N)
        :param sids: source node indices for the current batch
        :param position_ids: position indices for each node (N,)
        :param pos: positive node indices
        :param negs: negative node indices
        :return: InfoNCE loss
        """
        emb_s = self.emb_s  # (N, d1)
        emb_p = self.emb_p(position_ids)  # (N, d2)x
        x = torch.cat([emb_s, emb_p], dim=1)  # (N, d1 + d2)

        # Projection
        x = self.projection_layer(x)

        # GCN layers
        for idx, layer in enumerate(self.layers):
            if self.layers_name =="GCN":
                x_new = self.activation(layer(x))
                x_new = F.dropout(x_new, p=self.dropout)
                x_new = torch.sparse.mm(adj_norm, x_new)
            elif self.layers_name == "GAT":
                x_new = self.activation(layer(x, adj_norm))
                x_new = F.dropout(x_new, p=self.dropout)
            x = x + x_new
            self.embeddings_list[idx + 1] = x.detach().cpu()

        # Output
        out = self.output_layer(x)
        self.embeddings_list[-1] = out.detach().cpu()
        self.embeddings = torch.stack(self.embeddings_list).sum(dim=0).cpu()

        # Loss
        negs = negs.T
        batch_size, K = negs.shape

        emb_pos = out[sids]        # (batch, d)
        emb_pos_pair = out[pos]    # (batch, d)
        emb_neg = out[negs.reshape(-1)]
        emb_neg = emb_neg.view(batch_size, K, -1)
        # Cosine Similarity
        pos_sim = F.cosine_similarity(emb_pos, emb_pos_pair, dim=-1).unsqueeze(1)
        emb_pos = emb_pos.unsqueeze(1)
        neg_sim = F.cosine_similarity(emb_pos, emb_neg, dim=-1)

        # InfoNCE Loss
        loss_cl = -torch.log(torch.exp(pos_sim / self.temp) / (torch.exp(pos_sim / self.temp) + torch.exp(neg_sim / self.temp) + 1e-8))
        loss_cl = loss_cl.mean()

        # reg loss
        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()
        loss_reg = loss_reg * self.lambda_1
        loss = loss_cl + loss_reg

        return loss, loss_cl, loss_reg


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, bias=True):
        super().__init__()
        assert out_dim % heads == 0
        self.heads      = heads
        self.head_dim   = out_dim // heads
        self.W          = nn.Linear(in_dim, out_dim, bias=bias)
        self.a_src      = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_dst      = nn.Parameter(torch.empty(heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        # self.dropout    = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        if self.W.bias is not None: nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj_sp):
        Wh = self.W(h).view(-1, self.heads, self.head_dim)
        idx_i, idx_j = adj_sp._indices()

        outs = []
        for head in range(self.heads):
            Wh_h      = Wh[:, head, :]
            Wh_i, Wh_j = Wh_h[idx_i], Wh_h[idx_j]

            e_ij = (Wh_i * self.a_src[head]).sum(-1) + \
                   (Wh_j * self.a_dst[head]).sum(-1)
            e_ij = self.leaky_relu(e_ij)

            # sparse softmax
            att_h = torch.sparse_coo_tensor(
                        adj_sp._indices(), e_ij, adj_sp.shape,
                        device=h.device, dtype=e_ij.dtype)
            att_h = torch.sparse.softmax(att_h, dim=1)
            # att_h = self.dropout(att_h)

            # 메시지 집계
            h_prime_h = torch.sparse.mm(att_h, Wh_h)
            outs.append(h_prime_h)

        return torch.cat(outs, dim=1)


def sparse_attention_agg(adj_sp, src_emb, dst_emb,
                         q_lin, k_lin, leaky_relu, dropout):
    adj_sp = adj_sp.coalesce()
    idx = adj_sp.indices()                 # (2,E)
    row, col = idx[0], idx[1]

    q = q_lin(src_emb[row])                # (E,d)
    k = k_lin(dst_emb[col])
    e = (q * k).sum(-1) / math.sqrt(q.size(-1))
    e = leaky_relu(e)

    # ----- row‑wise softmax-----
    row_max = torch.zeros(src_emb.size(0), device=e.device)
    row_max.scatter_reduce_(0, row, e, reduce="amax", include_self=False)
    e = e - row_max[row]

    exp_e = torch.exp(e)
    row_sum = torch.zeros_like(row_max)
    row_sum.scatter_add_(0, row, exp_e)
    alpha = exp_e / (row_sum[row] + 1e-12)

    alpha = F.dropout(alpha, p=dropout, training=q_lin.training)

    # ----- sparse mm -----
    att_values = adj_sp.values() * alpha
    att_adj = torch.sparse_coo_tensor(idx, att_values,
                                      adj_sp.shape).coalesce()
    return torch.sparse.mm(att_adj, dst_emb)

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, device):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Linear(input_dim, 1).to(device)

    def forward(self, x):  # x: [num_skills, dim]
        # attn_scores: [num_skills, 1] → softmax → [num_skills, 1]
        attn_scores = F.softmax(self.attn(x), dim=0)
        # weighted sum: [dim]
        pooled = torch.sum(attn_scores * x, dim=0)
        return pooled

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4, device="cuda"):
        super(MultiHeadAttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.attn = nn.Linear(input_dim, num_heads, bias=False).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D]
        # attn_scores: [N, heads]
        attn_scores = self.attn(x)                    # (N, H)
        attn_scores = F.softmax(attn_scores, dim=0)   # softmax over skills per head

        # (attn_scores.unsqueeze(-1) * x.unsqueeze(1)) → [N, H, D]
        weighted = (attn_scores.unsqueeze(-1) * x.unsqueeze(1)).sum(dim=0)
        # weighted: (H, D)
        pooled = weighted.mean(dim=0)
        return pooled

class SumPooling(nn.Module):
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, x):
        return torch.sum(x, dim=0)

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, x):
        return x.mean(dim=0)

