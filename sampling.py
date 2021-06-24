import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import scipy
import sklearn
import pickle
import random
from tqdm import tqdm

def get_posneg_samples(tensor, exp, lookup_path, shRNA_dict_path, seed_sim_path, samples):
    
    batch = tensor[0].shape[0]
    lookup = pd.read_csv(lookup_path)
    with open(shRNA_dict_path, 'rb') as f:
        shRNA_dict = pickle.load(f)
    with open(seed_sim_path, 'rb') as f:
        seed_similarity = pickle.load(f)
    
    shRNA_idx_indata = []
    unique_shRNA = list(set(lookup['pert_mfc_id'].tolist()))
    for i in range(len(unique_shRNA)):
        shRNA_idx_indata.append(shRNA_dict[unique_shRNA[i]])                        
        
    pos = torch.zeros([batch*samples, tensor[0].shape[1]])
    neg = torch.zeros([batch*samples, tensor[0].shape[1]])
    for i in tqdm(range(batch)):
        anchor_target = tensor[4][i]
        anchor_shRNA = tensor[2][i]
        while True:
            pos_idx = random.sample(lookup[lookup['cmap_name'] == anchor_target].index.tolist(), samples)
            if tensor[5] not in pos_idx:
                break
        
        idx_shRNA = shRNA_dict[anchor_shRNA]
        while True:
            idx_shRNA_neg = torch.multinomial(torch.tensor(seed_similarity[idx_shRNA,:]+7), samples)
            if idx_shRNA not in list(idx_shRNA_neg) and set(idx_shRNA_neg.tolist()) & set(shRNA_idx_indata) == set(idx_shRNA_neg.tolist()):
                break
        k = 0
        for j in range(samples):
            pos[i+k] = torch.tensor(exp[pos_idx[j]])
            shRNA_neg = [shRNA for shRNA, value in shRNA_dict.items() if value == idx_shRNA_neg[j]][0]
            neg_idx = random.sample(lookup[lookup['pert_mfc_id'] == shRNA_neg].index.tolist(),1)
            neg[i+k] = torch.tensor(exp[neg_idx])
            k = k+batch
        
    return pos, neg