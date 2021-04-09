import numpy as np
import torch
import torch.utils.data as data

class Dataset_AE(data.Dataset):
    def __init__(self, profile):
        self.profile = profile
    
    def __getitem(self,index):
        src_profile = self.profile[index]
        trg_profile = self.profile[index]
        src_profile = torch.Tensor(src_profile)
        trg_profile = torch.Tensor(trg_profile)
        assert src_profile == trg_profile
        
        return src_profile, trg_profile
    
def collate_fn(data):
    source,target = zip(*data)
    src = merge(source)
    trg = merge(target)
    
    return src, trg

def merge(profiles,batch_size):
    profile = torch.zeros(batch_size, 978).long()
    for i, pf in enumerate(profiles):
        profile[i,:] = pf
    
    return profile

def get_loader(dataset, batch_size):
    data_loader = torch.utils.data.DataLodaer(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False
                                              collate_fn=collate_fn)
    return data_loader