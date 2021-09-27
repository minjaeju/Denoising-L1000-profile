import torch.nn as nn
import torch.nn.functional as F
import torch

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, pos, neg, size_average=False):
        #distance_positive = F.cosine_similarity(anchor, pos)
        #distance_negative = F.cosine_similarity(anchor, neg)
        distance_positive = F.cosine_similarity(anchor.unsqueeze(1), pos, dim=-1)
        #a = F.cosine_similarity(anchor.unsqueeze(1), neg, dim=-1)
        distance_negative = torch.abs(F.cosine_similarity(anchor.unsqueeze(1), neg, dim=-1))
        #import pdb; pdb.set_trace()
        #losses = F.relu(-(distance_positive.unsqueeze(1) - distance_negative.unsqueeze(1) - self.margin))
        losses = F.relu(-(distance_positive - distance_negative - self.margin))
        #import pdb; pdb.set_trace()
        return losses.mean() if size_average else losses.mean(0).sum()
