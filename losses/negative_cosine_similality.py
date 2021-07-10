import torch
import torch.nn as nn
import torch.nn.functional as F 

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class CosineContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z1, z2, p1, p2):
        if z1.dim() != 2:
            z1 = z1.squeeze()
        if z2.dim() != 2:
            z2 = z2.squeeze()

        if p1 is not None or p2 is not None:
            loss = D(p1, z2) / 2 + D(p2, z1) / 2
        else:
            loss = D(z1, z2)

        return loss