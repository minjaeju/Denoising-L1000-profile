import torch

import numpy as np
import pandas as pd
import scipy
import sklearn
import pickle
import random
from tqdm import tqdm

import pdb

def get_posneg_samples(tensor, exp, lookup_path, shRNA_dict_path, seed_sim_path, seed_dict_path, samples):
    
    batch = tensor[0].shape[0]
    lookup = pd.read_csv(lookup_path)
    with open(shRNA_dict_path, 'rb') as f:
        shRNA_dict = pickle.load(f)
    with open(seed_sim_path, 'rb') as f:
        seed_similarity = pickle.load(f)
    with open(seed_dict_path, 'rb') as f:
        seed_dict = pickle.load(f)
        
    shRNA_idx_indata = []
    unique_shRNA = list(set(lookup['pert_mfc_id'].tolist()))
    for i in range(len(unique_shRNA)):
        shRNA_idx_indata.append(shRNA_dict[unique_shRNA[i]])                        
        
    pos = torch.zeros([batch, samples, tensor[0].shape[1]])
    neg = torch.zeros([batch, samples, tensor[0].shape[1]])
    for i in tqdm(range(batch)):
        anchor_target = tensor[4][i]
        anchor_shRNA = tensor[2][i]
        if len(lookup[lookup['cmap_name'] == anchor_target].index.tolist()) <= samples:
            pos_idx = lookup[lookup['cmap_name'] == anchor_target].index.tolist()

            if len(pos_idx) != samples:
                while True:
                    pos_idx.append(tensor[5][i])
                    if len(pos_idx) == samples:
                        break
        else:
            while True:
                pos_idx = random.sample(lookup[lookup['cmap_name'] == anchor_target].index.tolist(), samples)
                if tensor[5][i] not in pos_idx:
                    break
        pos[i] = torch.tensor(exp[pos_idx])
        
        """for sampling by similarity
        idx_shRNA = shRNA_dict[anchor_shRNA]        
        
        while True:
            idx_shRNA_neg = torch.multinomial(torch.tensor(seed_similarity[idx_shRNA,:]+7), samples)
            if idx_shRNA not in list(idx_shRNA_neg) and set(idx_shRNA_neg.tolist()) & set(shRNA_idx_indata) == set(idx_shRNA_neg.tolist()):
                break
        shRNA_neg = [shRNA for shRNA, value in shRNA_dict.items() if value in idx_shRNA_neg]
        neg_idx = random.sample(lookup[lookup['pert_mfc_id'].isin (shRNA_neg)].index.tolist(),samples)
        """
        
        #print('pos_idx:', pos_idx)
        #print('pos_exp:', exp[pos_idx].shape)
                
        idx_shRNA = shRNA_dict[anchor_shRNA] 
        same_seed = seed_dict[anchor_shRNA]
        shRNA_neg = [shRNA for shRNA, value in shRNA_dict.items() if value in same_seed]
        if len(lookup[lookup['pert_mfc_id'].isin (shRNA_neg)].index.tolist()) == 0:
            while True:
                idx_shRNA_neg = torch.multinomial(torch.tensor(seed_similarity[idx_shRNA,:]+7), samples)
                if idx_shRNA not in list(idx_shRNA_neg) and set(idx_shRNA_neg.tolist()) & set(shRNA_idx_indata) == set(idx_shRNA_neg.tolist()):
                    break
            shRNA_neg = [shRNA for shRNA, value in shRNA_dict.items() if value in idx_shRNA_neg]
            neg_idx = random.sample(lookup[lookup['pert_mfc_id'].isin (shRNA_neg)].index.tolist(),samples)
           
        else:
            if len(lookup[lookup['pert_mfc_id'].isin (shRNA_neg)].index.tolist()) < samples:
                neg_idx = lookup[lookup['pert_mfc_id'].isin (shRNA_neg)].index.tolist()
                while True:
                    neg_idx.append(neg_idx[0])
                    if len(neg_idx) == samples:
                        break
            else:
                neg_idx = random.sample(lookup[lookup['pert_mfc_id'].isin (shRNA_neg)].index.tolist(),samples)
        #if len(neg_idx) == 2:
        #    pdb.set_trace()
        #print('index:', i)
        #print('neg_idx:', neg_idx)
        #print('pos_idx:', pos_idx)
        #print('neg_exp:', exp[neg_idx].shape)
        neg[i] = torch.tensor(exp[neg_idx])
        
    return pos, neg