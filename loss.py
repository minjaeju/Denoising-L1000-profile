import torch.nn as nn
import torch.nn.functional as F
import torch

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, pos, neg, size_average=True):
        distance_positive = F.cosine_similarity(anchor, pos)
        distance_negative = F.cosine_similarity(anchor, neg)
        #distance_positive = F.cosine_similarity(anchor.unsqueeze(1), pos, dim=2)
        #distance_negative = F.cosine_similarity(anchor.unsqueeze(1), neg, dim=2)
        #losses = F.relu(-(distance_positive.unsqueeze(1) - distance_negative.unsqueeze(1) - self.margin))
        losses = F.relu(-(distance_positive - distance_negative - self.margin))

        return losses.mean() if size_average else losses.sum()