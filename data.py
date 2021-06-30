import numpy as np
import torch
import torch.utils.data as data

class Dataset_AE(data.Dataset):
    
    def __init__(self, profile, lookup_table):
        self.profile = profile
        self.table = lookup_table
        
    def __len__(self):
        return len(self.profile)
        
    def __getitem__(self, index):
        src_profile = self.profile[index]
        trg_profile = self.profile[index]
        shRNA = self.table.iloc[index]['pert_mfc_id']
        cell = self.table.iloc[index]['cell_mfc_name']
        gene = self.table.iloc[index]['cmap_name']
        #print(index)
        assert np.array_equal(src_profile, trg_profile) == True
        
        return src_profile, trg_profile, shRNA, cell, gene, index       
    
def collate_fn(data):
    tensor_list = []
    src = [x[0] for x in data]
    trg = [x[1] for x in data]
    shrna = [x[2] for x in data]
    cell = [x[3] for x in data]
    gene = [x[4] for x in data]
    index = [x[5] for x in data]
    tensor_list.append(torch.cuda.FloatTensor(src))
    tensor_list.append(torch.cuda.FloatTensor(trg))
    tensor_list.append(shrna)
    tensor_list.append(cell)
    tensor_list.append(gene)
    tensor_list.append(index)

    return tensor_list

def get_loader(dataset, sampler, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              sampler=sampler)
    return data_loader

def get_loader_infer(dataset, batch_size):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn)

    return data_loader