import timm

import torch.nn as nn

class NFNet(nn.Module):
    def __init__(self, num_classes, arch='nfnet_f0'):
        super().__init__()
        m = timm.create_model(arch, pretrained=False)
        self.feature = nn.Sequential(*list(m.children())[:-1])
        self.feature.add_module('global_pool', m.head.global_pool)

        self.fc = nn.Linear(3072, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.squeeze()
        out = self.fc(x)
        return out