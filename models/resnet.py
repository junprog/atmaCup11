import torch.nn as nn

import  torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, arch, num_classes):
        super().__init__()
        m = models.__dict__[arch](pretrained=False)
        self.feature = nn.Sequential(*list(m.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.squeeze()
        out = self.fc(x)
        return out