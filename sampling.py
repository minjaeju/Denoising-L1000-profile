import torch

import numpy as np
import pandas as pd
import scipy
import sklearn
import pickle
import random
from tqdm import tqdm

import pdb

def get_posneg_samples(tensor, exp, lookup_df, shRNA_dict_path, seed_sim_path, seed_dict_path, samples):
    
    batch = tensor[0].shape[0]
    lookup = lookup_df
    with open(shRNA_dict_path, 'rb') as f:
        shRNA_dict = pickle.load(f)
    with open(seed_sim_path, 'rb') as f:
        seed_similarity = pickle.load(f)
    with open(seed_dict_path, 'rb') as f:
        seed_dict = pickle.load(f)
        
    shRNA_idx_indata = set()
    unique_shRNA = list(set(lookup['pert_mfc_id'].tolist()))
    for i in range(len(unique_shRNA)):
        shRNA_idx_indata.add(shRNA_dict[unique_shRNA[i]])
        
    pos = torch.zeros([batch, samples, tensor[0].shape[1]])
    neg = torch.zeros([batch, samples, tensor[0].shape[1]])
    for i in range(batch):
        anchor_target = tensor[4][i]
        anchor_shRNA = tensor[2][i]
        
        #pos
        pidx = lookup[lookup['cmap_name'] == anchor_target].index.tolist()
        pidx.remove(tensor[5][i])
        pidx = pidx + [tensor[5][i] for j in range(samples - len(pidx))]
        pos_idx = random.sample(pidx, samples)
        
        pos[i] = torch.tensor(exp[pos_idx])
        
        #neg
        idx_shRNA = shRNA_dict[anchor_shRNA] 
        same_seed = seed_dict[anchor_shRNA]
        shRNA_neg = [shRNA for shRNA, value in shRNA_dict.items() if value in same_seed]
        if len(lookup[lookup['pert_mfc_id'].isin (shRNA_neg) & (lookup['cmap_name'] != anchor_target)].index.tolist()) == 0:
            idx_shRNA_neg = torch.multinomial(torch.tensor(seed_similarity[idx_shRNA,:]+7), samples).tolist()
            idx_shRNA_neg = set(idx_shRNA_neg) & (shRNA_idx_indata - {idx_shRNA})
            shRNA_neg = [shRNA for shRNA, value in shRNA_dict.items() if value in idx_shRNA_neg]

            nidx = lookup[(lookup['pert_mfc_id'].isin (shRNA_neg)) & (lookup['cmap_name'] != anchor_target)].index.tolist()
            
            if len(nidx) == 0:
                neg_idx = random.sample(lookup[lookup['cmap_name'] != anchor_target].index.tolist(),samples)
                pdb.set_trace()
            else:
                nidx = nidx + [nidx[0] for j in range(samples - len(nidx))]
                neg_idx = random.sample(nidx,samples)
           
        else:
            nidx = lookup[(lookup['pert_mfc_id'].isin (shRNA_neg)) & (lookup['cmap_name'] != anchor_target)].index.tolist()
            nidx = nidx + [nidx[0] for j in range(samples - len(nidx))]
            neg_idx = random.sample(nidx,samples)

        neg[i] = torch.tensor(exp[neg_idx])

    return pos, neg

"""
while True:
                idx_shRNA_neg = torch.multinomial(torch.tensor(seed_similarity[idx_shRNA,:]+7), samples)
                if idx_shRNA not in list(idx_shRNA_neg) and set(idx_shRNA_neg.tolist()) & set(shRNA_idx_indata) == set(idx_shRNA_neg.tolist()):
                    break
            shRNA_neg = [shRNA for shRNA, value in shRNA_dict.items() if value in idx_shRNA_neg]
            neg_idx = random.sample(lookup[lookup['pert_mfc_id'].isin (shRNA_neg)].index.tolist(),samples)
"""