import sys
import os
current_path = os.getcwd()
sys.path.append(current_path) # カレントディレクトリをパスに追加

import torch
import torch.nn as nn

from collections import OrderedDict

from models.resnet import ResNet
from models.efficient_net import EfficientNet

# from vit_pytorch import ViT

class DoubleNet(nn.Module):
    def __init__(self, mate_classes=6, tech_classes=3, resnet_weight_path='', efnet_weight_path='', freeze=True):
        super().__init__()

        self.resnet = ResNet('resnet34', 1).feature
        self.efnet = EfficientNet('efficientnet_b0', 1).feature

        self._init_weight(self.resnet, resnet_weight_path)
        self._init_weight(self.efnet, efnet_weight_path)

        if freeze:
            self._freeze(self.resnet)
            self._freeze(self.efnet)

        self.fc = nn.Linear(512 + 1280, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        self.reg1 = nn.Linear(1024, 512)
        self.reg2 = nn.Linear(512, 1)

        self.mate1 = nn.Linear(1024, 512)
        self.mate2 = nn.Linear(512, mate_classes)

        self.tech1 = nn.Linear(1024, 512)
        self.tech2 = nn.Linear(512, tech_classes)
        
    def forward(self, x):
        res_x, ef_x = self.resnet(x), self.efnet(x)
        res_x, ef_x = res_x.squeeze(), ef_x.squeeze()

        if res_x.dim() == 1:
            res_x = res_x.unsqueeze(0)
        if ef_x.dim() == 1:
            ef_x = ef_x.unsqueeze(0)

        fusion_x = torch.cat([res_x, ef_x], dim=1)

        x = self.fc(fusion_x)
        x = self.relu(x)

        reg = self.reg1(x)
        reg = self.relu(reg)
        reg = self.dropout(reg)
        reg = self.reg2(reg)

        mate = self.mate1(x)
        mate = self.relu(mate)
        mate = self.dropout(mate)
        mate = self.mate2(mate)

        tech = self.tech1(x)
        tech = self.relu(tech)
        tech = self.dropout(tech)
        tech = self.tech2(tech)

        return reg, mate, tech

    def _init_weight(self, model, weight_path):
        if weight_path:
            suf = weight_path.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(weight_path)
                new_checkpoint = OrderedDict()
                
                for saved_key, saved_value in checkpoint['model_state_dict'].items():
                    if 'projector' in saved_key or 'predictor' in saved_key or 'fc' in saved_key:
                        continue
                    else:
                        saved_key = saved_key.replace('encoder.', '')
                        saved_key = saved_key.replace('feature.', '')
                        new_checkpoint[saved_key] = saved_value

                model.load_state_dict(new_checkpoint)

            elif suf == 'pth':
                checkpoint = torch.load(weight_path)
                new_checkpoint = OrderedDict()
                
                for saved_key, saved_value in checkpoint.items():
                    if 'projector' in saved_key or 'predictor' in saved_key or 'fc' in saved_key:
                        continue
                    else:
                        if 'reg' in saved_key or 'mate' in saved_key or 'tech' in saved_key:
                            continue
                        saved_key = saved_key.replace('encoder.', '')
                        saved_key = saved_key.replace('feature.', '')
                        new_checkpoint[saved_key] = saved_value

                model.load_state_dict(new_checkpoint)

    def _freeze(self, model):
        for params in model.parameters():
            params.requires_grad = False

    def unfreeze(self):
        for params in self.parameters():
            params.requires_grad = True

if __name__ == '__main__':
    inputs = torch.rand(4, 3, 224, 224)
    model = DoubleNet()

    print(model)
    reg, mate, tech = model(inputs)
    print(reg.shape, mate.shape, tech.shape)
    model.unfreeze()