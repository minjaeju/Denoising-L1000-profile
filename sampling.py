import torch

import numpy as np
import pandas as pd
import scipy
import sklearn
import pickle
import random
from tqdm import tqdm

import pdb

def get_neg_dict(lookup_df, samples, test=None):
    
    #for {shRNA: neg_sample}
    lookup = lookup_df
    
    shRNA = pd.read_csv('./data/shRNA/shRNA_processed.csv')

    with open('./data/shRNA/seed_dict_neg.pkl', 'rb') as f:
        seed_dict = pickle.load(f)
            
    neg = {}
    for i in tqdm(range(len(shRNA))):
        anchor_target = shRNA.symbol[i]
        anchor_shRNA = shRNA.cloneId[i]
        same_seed = seed_dict[anchor_shRNA]
        if len(lookup[(lookup['pert_mfc_id'].isin (same_seed))]) == 0:
            diff_target = [shRNA.cloneId[j] for j in range(len(shRNA)) if shRNA.symbol[j] != anchor_target]
            nidx = lookup[(lookup['pert_mfc_id'].isin (diff_target))].index.tolist()
        else:
            nidx = lookup[(lookup['pert_mfc_id'].isin (same_seed))].index.tolist()
        nidx = nidx + [nidx[0] for j in range(samples - len(nidx))]
        
        neg[shRNA.cloneId[i]] = nidx
    if test:
        with open('./data/neg_dict_test.pkl', 'wb') as f:
            pickle.dump(neg, f)
    else:
        with open('./data/neg_dict.pkl', 'wb') as f:
            pickle.dump(neg, f)
    print('negative index dictionary completed')


def get_posneg_samples(tensor, exp, lookup_df, samples, neg_dic, test=None):
    
    lookup = lookup_df
    #neg_dict exists   
    batch = tensor[0].shape[0]        
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
        neg_idx = random.sample(neg_dic[anchor_shRNA], samples)
        neg[i] = torch.tensor(exp[neg_idx])

    return pos, neg