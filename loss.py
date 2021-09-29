import torch.nn as nn
import torch.nn.functional as F
import torch

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, pos, neg, size_average=False):
        distance_positive = F.cosine_similarity(anchor.unsqueeze(1), pos, dim=-1)
        distance_negative = torch.abs(F.cosine_similarity(anchor.unsqueeze(1), neg, dim=-1))
        losses = F.relu(-(distance_positive - distance_negative - self.margin))

        return losses.mean() if size_average else losses.mean(0).sum()
