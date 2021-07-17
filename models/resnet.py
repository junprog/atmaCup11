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

class MultiBranchResNet(nn.Module):
    def __init__(self, arch, num_classes ,mate_classes, tech_classes):
        super().__init__()
        m = models.__dict__[arch](pretrained=False)
        self.feature = nn.Sequential(*list(m.children())[:-1])

        self.relu = nn.ReLU(inplace=True)
        self.reg1 = nn.Linear(512, 256)
        self.reg2 = nn.Linear(256, num_classes)

        self.mate1 = nn.Linear(512, 256)
        self.mate2 = nn.Linear(256, mate_classes)

        self.tech1 = nn.Linear(512, 256)
        self.tech2 = nn.Linear(256, tech_classes)

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

class MultiBranchResNet_v2(nn.Module):
    def __init__(self, arch, num_classes ,mate_classes, tech_classes):
        super().__init__()
        m = models.__dict__[arch](pretrained=False)
        self.feature = nn.Sequential(*list(m.children())[:-1])

        self.relu = nn.ReLU(inplace=True)
        self.reg1 = nn.Linear(512, 256)
        self.reg2_hard = nn.Linear(256, 1)
        self.reg2_soft = nn.Linear(256, 1)
        self.cls = nn.Linear(256, num_classes)

        self.mate1 = nn.Linear(512, 256)
        self.mate2 = nn.Linear(256, mate_classes)

        self.tech1 = nn.Linear(512, 256)
        self.tech2 = nn.Linear(256, tech_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.squeeze()

        reg = self.reg1(x)
        reg = self.relu(reg)
        reg_hard = self.reg2_hard(reg)
        reg_soft = self.reg2_soft(reg)
        logit = self.cls(reg)

        mate = self.mate1(x)
        mate = self.relu(mate)
        mate = self.mate2(mate)

        tech = self.tech1(x)
        tech = self.relu(tech)
        tech = self.tech2(tech)

        return reg_hard, logit, reg_soft, mate, tech