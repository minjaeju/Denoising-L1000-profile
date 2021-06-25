import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_hidden_he(layer):
    layer.apply(init_relu)

def init_relu(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, 2 ** 0.5)

class VanillaEncoder(nn.Module):
    
    def __init__(self):
        super(VanillaEncoder, self).__init__()
        self.num_layer = 3
        self.hidden_dim = [978, 512, 256, 256]
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.MLP = nn.ModuleList(
            [nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]) for i in range(self.num_layer-1)])
        self.MLP_ft = nn.Linear(256,256)

    def forward(self, exp, pos=None, neg=None, finetune = False):
        if finetune == True:
            for i in range(self.num_layer-1):
                exp = self.dropout(self.activation(self.MLP[i](exp)))
                pos = self.dropout(self.activation(self.MLP[i](pos)))
                neg = self.dropout(self.activation(self.MLP[i](neg)))
                
            return self.MLP_ft(exp), self.MLP_ft(pos), self.MLP_ft(neg)
        
        else:
            for i in range(self.num_layer-1):
                exp = self.dropout(self.activation(self.MLP[i](exp)))
            
            return self.MLP_ft(exp)

class VanillaDecoder(nn.Module):

    def __init__(self):

        super(VanillaDecoder, self).__init__()

        self.num_layer = 3
        self.hidden_dim = [256, 256, 512, 978]
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.MLP = nn.ModuleList(
            [nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]) for i in range(self.num_layer)])
        init_hidden_he(self.MLP)

    def forward(self, exp):

        for i in range(self.num_layer):
            if i != self.num_layer - 1:
                exp = self.dropout(self.activation(self.MLP[i](exp)))

            else:
                exp = self.MLP[i](exp)

        return exp

class VanillaAE(nn.Module):

    def __init__(self):
        super(VanillaAE, self).__init__()

        #self.profile = profile
        self.encoder = VanillaEncoder()
        self.decoder = VanillaDecoder()
        
    def forward(self, profile):

        profile_embed = self.encoder(profile)
        profile_recon = self.decoder(profile_embed)

        return profile_recon