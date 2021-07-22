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

class MultiBranchNFNet(nn.Module):
    def __init__(self, arch, num_classes ,mate_classes, tech_classes):
        super().__init__()
        m = timm.create_model(arch, pretrained=False)
        self.feature = nn.Sequential(*list(m.children())[:-1])
        self.feature.add_module('global_pool', m.head.global_pool)

        self.relu = nn.ReLU(inplace=True)
        self.reg1 = nn.Linear(3072, 512)
        self.reg2 = nn.Linear(512, num_classes)

        self.mate1 = nn.Linear(3072, 512)
        self.mate2 = nn.Linear(512, mate_classes)

        self.tech1 = nn.Linear(3072, 512)
        self.tech2 = nn.Linear(512, tech_classes)

    def forward(self, x):
        x = self.feature(x)
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