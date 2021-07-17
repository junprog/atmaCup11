import timm

import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, num_classes, arch='vit_small_patch16_224'):
        super().__init__()
        m = timm.create_model(arch, pretrained=False)
        self.feature = nn.Sequential(*list(m.children())[:-1])
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.squeeze()
        out = self.fc(x)
        return out