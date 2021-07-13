import timm

import torch.nn as nn

class EfficientNet(nn.Module):
    def __init__(self, arch, num_classes):
        super().__init__()
        m = timm.create_model(arch, pretrained=False)
        self.feature = nn.Sequential(*list(m.children())[:-1])
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.squeeze()
        out = self.fc(x)
        return out
