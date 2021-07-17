import timm

import torch.nn as nn

class NFNet(nn.Module):
    def __init__(self, num_classes, arch='nfnet_f0'):
        super().__init__()
        m = timm.create_model(arch, pretrained=False)
        self.feature = nn.Sequential(*list(m.children())[:-1])
        self.fc = nn.Sequential(*list(m.children())[-1:])
        self.fc.head = nn.Linear(3072, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.squeeze()
        out = self.fc(x)
        return out