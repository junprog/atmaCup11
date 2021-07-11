import sys
import os
current_path = os.getcwd()
sys.path.append(current_path) # カレントディレクトリをパスに追加

import torch
import torch.nn as nn

from collections import OrderedDict

from models.resnet import ResNet

class FusionNet(nn.Module):
    def __init__(self, arch, simsiam_weight_path, mate_weight_path, tech_weight_path, freeze=True):
        super().__init__()

        self.main_net = ResNet(arch, 1).feature
        self.mate_net = ResNet(arch, 1).feature
        self.tech_net = ResNet(arch, 1).feature

        self._init_weight(self.main_net, simsiam_weight_path)
        self._init_weight(self.mate_net, mate_weight_path)
        self._init_weight(self.tech_net, tech_weight_path)

        if freeze:
            self._freeze(self.mate_net)
            self._freeze(self.tech_net)

        self.fc1 = nn.Linear(512*3, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, x):
        main_x, mate_x, tech_x = self.main_net(x), self.mate_net(x), self.tech_net(x)
        main_x, mate_x, tech_x = main_x.squeeze(), mate_x.squeeze(), tech_x.squeeze()

        if main_x.dim() == 1:
            main_x, mate_x, tech_x = main_x.unsqueeze(0), mate_x.unsqueeze(0), tech_x.unsqueeze(0)

        fusion_x = torch.cat([main_x, mate_x, tech_x], dim=1)

        x = self.fc1(fusion_x)
        x = self.relu(x)
        out = self.fc2(x)

        return out

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
                        saved_key = saved_key.replace('encoder.', '')
                        saved_key = saved_key.replace('feature.', '')
                        new_checkpoint[saved_key] = saved_value

                model.load_state_dict(new_checkpoint)

    def _freeze(self, model):
        for params in model.parameters():
            params.requires_grad = False

    def unfreeze(self):
        for params in self.parameters():
            params.requires_grad = False

if __name__ == '__main__':
    inputs = torch.rand(4, 3, 224, 224)

    model = FusionNet(
        arch='resnet34',
        simsiam_weight_path='simsiam_logs/exp02-0710-180339/300_ckpt.tar',
        mate_weight_path='material_logs/exp02-0711-221109/cv_0/0_ckpt.tar',
        tech_weight_path='technique_logs/exp02-0711-235630/cv_0/0_ckpt.tar',
        freeze=True
    )
    
    outputs = model(inputs)

    print(outputs.shape)

    model.unfreeze()