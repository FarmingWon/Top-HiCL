import numpy as np
import torch
import torch.nn as nn
import pickle
from models import Top_HiCL_H, Top_HiCL
from utils import HierarchySkillLoader, JobSkillLoader, augmented_mean_pooling_job_emb, coo_bipartite_normalized, coo_normalized
from parser_s1 import args
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyperparameters
gnn_layer = args.gnn_layer
temp = args.temp
epochs = args.epoch
dropout = args.dropout
lr = args.lr
weight_decay = args.weight_decay # L2 Regularization
lambda_1 = args.lambda1
save_path = os.path.join(os.getcwd(), "checkpoints/")
aug_dim = [[args.dim, args.dim] for i in range(gnn_layer)]

# hierarchy data load
hpath = os.path.join(os.getcwd(), args.hpath)
f = open(hpath,'rb')
hdatas = pickle.load(f) # arr: coo_sp, emb_j: bert_emb_jobs, emb_s : bert_emb_skills
h_train = hdatas['arr']
h_emb_s = torch.tensor(np.array(hdatas['emb_s']), dtype=torch.float32, device=device).squeeze(1)
h_depth = hdatas['depth']
h_parents = hdatas['parents']
adj_norm = coo_normalized(h_train, device) # adj coo_matrix normalized

# bipartite graph data load
path = os.path.join(os.getcwd(), args.path)
f = open(path,'rb')
datas = pickle.load(f) # arr: coo_sp, emb_j: emb_jobs, emb_s : emb_skills
train = datas['arr']
emb_j = datas['emb_j']
emb_s = datas['emb_s']
adj_row_norm, adj_col_norm = coo_bipartite_normalized(train, device)


# Construct data loader
h_train_data = HierarchySkillLoader(h_train.tocoo(), h_depth, h_parents, args.num_neg_samples)
h_train_loader  = torch.utils.data.DataLoader(h_train_data, batch_size = args.aug_batch, shuffle = True)
train = train.tocoo()
train_data = JobSkillLoader(train, args.num_neg_samples)
train_loader  = torch.utils.data.DataLoader(train_data, batch_size = args.batch, shuffle = True)
print('Data Loading...')


# # Activation
activation_h = nn.LeakyReLU(args.activation)
activation_o = nn.LeakyReLU(args.activation)

# ## model params: e_j_f, e_s_f, aug_e, num_layers, dim, aug_dim, temp, activation, dropout, bias, device
model_h = Top_HiCL_H(h_emb_s, args.pos_dim, args.dim, args.h_depth, gnn_layer, lambda_1, activation_h, temp, dropout, bias=True, device=device).to(device)
optimizer_h = torch.optim.Adam(model_h.parameters(), weight_decay=weight_decay, lr=lr)

model_o = Top_HiCL(emb_j, emb_s, gnn_layer, adj_row_norm, adj_col_norm, args.dim, temp, lambda_1, activation_o, dropout, bias=True, device=device).to(device)
optimizer_o = torch.optim.Adam(model_o.parameters(), weight_decay=weight_decay, lr=lr)    

depths = list()
for idx in range(len(h_depth)):
    depths.append(h_depth[idx])
depth = torch.tensor(depths).long().to(device)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_loss_cl = 0
    epoch_loss_reg = 0

    for i, batch in enumerate(tqdm(h_train_loader, desc="Step-1 Aug batch")):
        cols, pos, negs, depth_batch = batch
        cols = cols.long().to(device)
        pos = pos.long().to(device)
        negs = torch.stack(negs).long().to(device)

        optimizer_h.zero_grad()
        loss, loss_cl, loss_reg = model_h(adj_norm, cols, depth, pos, negs)

        loss.backward()
        optimizer_h.step()

        epoch_loss += loss.cpu().item()
        epoch_loss_cl += loss_cl.cpu().item()
        epoch_loss_reg += loss_reg.cpu().item()
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        aug_e_s = model_h.embeddings.detach()
        aug_e_j = augmented_mean_pooling_job_emb(train, aug_e_s, device)

    model_o.update_aug_embeddings(aug_e_j, aug_e_s, device)  

    for i, batch in enumerate(tqdm(train_loader, desc="Step-1 CL batch")):
        jids, sids, pos, negs = batch
        jids = jids.long().to(device)
        sids = sids.long().to(device)
        negs = [n.long().to(device) for n in negs]

        optimizer_o.zero_grad()
        loss, loss_cl, loss_reg = model_o(jids, sids, negs)
        loss.backward()
        optimizer_o.step()

        epoch_loss += loss.cpu().item()
        epoch_loss_cl += loss_cl.cpu().item()
        epoch_loss_reg += loss_reg.cpu().item()
        torch.cuda.empty_cache()
    
    if (epoch+1) % 100 == 0:
        torch.save({
            'model_state_dict': model_o.state_dict(),
            'optimizer_state_dict': optimizer_o.state_dict(),
            'epoch': epoch,
            'j_embeddings': model_o.E_j,
            's_embeddings': model_o.E_s
        }, os.path.join(save_path, f"{epoch+1}_checkpoints.pth"))
    
    print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_cl:',epoch_loss_cl,'Loss_reg:',epoch_loss_reg)