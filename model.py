import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VanillaAE(nn.Module):

    def __init__(self, profile):
        super(VanillaAE, self).__init__()

        self.profile = profile
        self.encoder = VanillaEncoder()
        self.decoder = VanillaDecoder()

    def forward(self, profile):

        profile_embed = self.encoder(profile)
        profile_recon = self.decoder(profile_embed)

        return profile_recon
    
class VanillaEncoder(nn.Module):
    
    def __init__(self):
        super(VanillaEncoder, self).__init__()
        self.num_layer = 2
        self.hidden_dim = [978, 512, 256]
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.MLP = nn.ModuleList(
            [nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]) for i in range(self.num_layer)])

    def forward(self, exp):

        for i in range(self.num_layer):

            if i != self.num_layer - 1:
                embedding = self.dropout(self.activation(self.MLP[i](exp)))

            else:
                embedding = self.MLP_profile[i](exp)

        return embedding

class VanillaDecoder(nn.Module):

    def __init__(self):

        super(VanillaDecoder, self).__init__()

        self.num_layer = 2
        self.hidden_dim = [256, 512, 978]
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.MLP = nn.ModuleList(
            [nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]) for i in range(self.num_layer)])
        init_hidden_he(self.MLP)


    def forward(self, embed):

        for i in range(self.num_layer):
            if i != self.num_layer - 1:
                exp = self.dropout(self.activation(self.MLP[i](embed)))

            else:
                exp = self.MLP[i](embed)
        return exp