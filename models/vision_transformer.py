import timm
import torch

import torch.nn as nn
from vit_pytorch import ViT

class VisionTransformer(nn.Module):
    def __init__(self, arch='vit_small_patch16_224'):
        super().__init__()
        self.vit = ViT(image_size=256, patch_size=16, num_classes=128, dim=512, depth=6, heads=16, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
        #self.feature = nn.Sequential(*list(m.children())[:-1])

    def forward(self, x):
        out = self.vit.forward_features(x)
        #x = x.squeeze()
        #out = self.fc(x)
        return out

class MultiBranchViT(nn.Module):
    def __init__(self, num_classes=1 ,mate_classes=6, tech_classes=3):
        super().__init__()
        self.vit = ViT(image_size=256, patch_size=16, num_classes=128, dim=512, depth=6, heads=16, mlp_dim=512, dropout=0.1, emb_dropout=0.1)

        self.relu = nn.ReLU(inplace=True)
        self.reg1 = nn.Linear(128, 64)
        self.reg2 = nn.Linear(64, num_classes)

        self.mate1 = nn.Linear(128, 64)
        self.mate2 = nn.Linear(64, mate_classes)

        self.tech1 = nn.Linear(128, 64)
        self.tech2 = nn.Linear(64, tech_classes)

    def forward(self, x):
        x = self.vit(x)
        x = x.squeeze()

        reg = self.reg1(x)
        reg = self.relu(reg)
        reg = self.reg2(reg)

        mate = self.mate1(x)
        mate = self.relu(mate)
        mate = self.mate2(mate)

        tech = self.tech1(x)
        tech = self.relu(tech)
        tech = self.tech2(tech)

        return reg, mate, tech

if __name__ == '__main__':
    inputs = torch.rand(4, 3, 256, 256)
    model = MultiBranchViT()
    out, mate, tech = model(inputs)

    print(out.shape, mate.shape, tech.shape)

    