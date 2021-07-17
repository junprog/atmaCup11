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

class MultiBranchEfficientNet(nn.Module):
    def __init__(self, arch, num_classes ,mate_classes, tech_classes):
        super().__init__()
        m = timm.create_model(arch, pretrained=False)
        self.feature = nn.Sequential(*list(m.children())[:-1])

        self.relu = nn.ReLU(inplace=True)
        self.reg1 = nn.Linear(1280, 512)
        self.reg2 = nn.Linear(512, num_classes)

        self.mate1 = nn.Linear(1280, 512)
        self.mate2 = nn.Linear(512, mate_classes)

        self.tech1 = nn.Linear(1280, 512)
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

class MultiBranchEfficientNet_cls(nn.Module):
    def __init__(self, arch, num_classes ,mate_classes, tech_classes):
        super().__init__()
        m = timm.create_model(arch, pretrained=False)
        self.feature = nn.Sequential(*list(m.children())[:-1])

        self.relu = nn.ReLU(inplace=True)
        self.cls1 = nn.Linear(1280, 512)
        self.cls2 = nn.Linear(512, num_classes)

        self.mate1 = nn.Linear(1280, 512)
        self.mate2 = nn.Linear(512, mate_classes)

        self.tech1 = nn.Linear(1280, 512)
        self.tech2 = nn.Linear(512, tech_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.squeeze()

        lgt = self.cls1(x)
        lgt = self.relu(lgt)
        lgt = self.cls2(lgt)

        mate = self.mate1(x)
        mate = self.relu(mate)
        mate = self.mate2(mate)

        tech = self.tech1(x)
        tech = self.relu(tech)
        tech = self.tech2(tech)

        return lgt, mate, tech

class MultiBranchEfficientNet_v2(nn.Module):
    def __init__(self, arch, num_classes ,mate_classes, tech_classes):
        super().__init__()
        m = timm.create_model(arch, pretrained=False)
        self.feature = nn.Sequential(*list(m.children())[:-1])

        self.relu = nn.ReLU(inplace=True)
        self.reg1 = nn.Linear(1280, 512)
        self.reg2_hard = nn.Linear(512, 1)
        self.reg2_soft = nn.Linear(512, 1)
        self.cls = nn.Linear(512, num_classes)

        self.mate1 = nn.Linear(1280, 512)
        self.mate2 = nn.Linear(512, mate_classes)

        self.tech1 = nn.Linear(1280, 512)
        self.tech2 = nn.Linear(512, tech_classes)

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

class MultiBranchEfficientNet_v3(nn.Module):
    def __init__(self, arch, num_classes ,mate_classes, tech_classes):
        super().__init__()
        m = timm.create_model(arch, pretrained=False)
        self.feature = nn.Sequential(*list(m.children())[:-1])

        self.relu = nn.ReLU(inplace=True)
        self.hidden = nn.Linear(1280, 512)
        self.reg = nn.Linear(512, 1)
        self.cls = nn.Linear(512, num_classes)

        self.mate1 = nn.Linear(1280, 512)
        self.mate2 = nn.Linear(512, mate_classes)

        self.tech1 = nn.Linear(1280, 512)
        self.tech2 = nn.Linear(512, tech_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.squeeze()

        x_hid = self.hidden(x)
        x_hid = self.relu(x_hid)
        reg = self.reg(x_hid)
        logit = self.cls(x_hid)

        mate = self.mate1(x)
        mate = self.relu(mate)
        mate = self.mate2(mate)

        tech = self.tech1(x)
        tech = self.relu(tech)
        tech = self.tech2(tech)

        return reg, logit, mate, tech