import sys
import os
current_path = os.getcwd()
sys.path.append(current_path) # カレントディレクトリをパスに追加

import torch
import torch.nn as nn

from collections import OrderedDict

from models.resnet import ResNet
from models.efficient_net import EfficientNet

class FcFusionNet_v1(nn.Module):
    def __init__(self, arch, simsiam_weight_path='', mate_weight_path='', tech_weight_path='', freeze=True):
        super().__init__()

        if 'resnet' in arch:
            self.main_net = ResNet(arch, 1).feature
            self.mate_net = ResNet(arch, 1).feature
            self.tech_net = ResNet(arch, 1).feature
        elif 'efficientnet' in arch:
            self.main_net = EfficientNet(arch, 1).feature
            self.mate_net = EfficientNet(arch, 1).feature
            self.tech_net = EfficientNet(arch, 1).feature

        self._init_weight(self.main_net, simsiam_weight_path)
        self._init_weight(self.mate_net, mate_weight_path)
        self._init_weight(self.tech_net, tech_weight_path)

        if freeze:
            self._freeze(self.mate_net)
            self._freeze(self.tech_net)

        if 'resnet' in arch:
            self.fc1 = nn.Linear(512*3, 512)
        elif 'efficientnet' in arch:
            self.fc1 = nn.Linear(1280*3, 512)

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
            params.requires_grad = True

class FcFusionNet_v2(nn.Module):
    def __init__(self, arch, mate_out_num, tech_out_num, simsiam_weight_path='', mate_weight_path='', tech_weight_path='', freeze=True):
        super().__init__()

        if 'resnet' in arch:
            self.main_net = ResNet(arch, 1).feature
            self.mate_net = ResNet(arch, mate_out_num)
            self.tech_net = ResNet(arch, tech_out_num)
        elif 'efficientnet' in arch:
            self.main_net = EfficientNet(arch, 1).feature
            self.mate_net = EfficientNet(arch, mate_out_num)
            self.tech_net = EfficientNet(arch, tech_out_num)

        self._init_weight(self.main_net, simsiam_weight_path)
        self._init_weight(self.mate_net, mate_weight_path)
        self._init_weight(self.tech_net, tech_weight_path)

        if freeze:
            self._freeze(self.mate_net)
            self._freeze(self.tech_net)

        if 'resnet' in arch:
            self.fc1 = nn.Linear(512 + mate_out_num + tech_out_num, 256)
        elif 'efficientnet' in arch:
            self.fc1 = nn.Linear(1280 + mate_out_num + tech_out_num, 256)

        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 1)
        
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
                model.load_state_dict(checkpoint)

    def _freeze(self, model):
        for params in model.parameters():
            params.requires_grad = False

    def unfreeze(self):
        for params in self.parameters():
            params.requires_grad = True

class Conv1dFusionNet(nn.Module):
    def __init__(self, arch, simsiam_weight_path='', mate_weight_path='', tech_weight_path='', freeze=True):
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

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(1024, 2048, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        
    def forward(self, x):
        main_x, mate_x, tech_x = self.main_net(x), self.mate_net(x), self.tech_net(x)
        main_x, mate_x, tech_x = main_x.squeeze(), mate_x.squeeze(), tech_x.squeeze()

        # N, C
        if main_x.dim() == 1:
            main_x, mate_x, tech_x = main_x.unsqueeze(0), mate_x.unsqueeze(0), tech_x.unsqueeze(0)

        # N, C, models
        fusion_x = torch.stack([main_x, mate_x, tech_x], dim=2)

        x = self.conv1(fusion_x)
        x = self.relu(x)
        x = self.conv2(x)

        x = x.squeeze()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        out = self.fc3(x)

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
            params.requires_grad = True

class Conv2dFusionNet(nn.Module):
    def __init__(self, arch, simsiam_weight_path='', mate_weight_path='', tech_weight_path='', freeze=True):
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

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 3), stride=1, padding=(0, 0))

        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 1)
        
    def forward(self, x):
        main_x, mate_x, tech_x = self.main_net(x), self.mate_net(x), self.tech_net(x)
        main_x, mate_x, tech_x = main_x.squeeze(), mate_x.squeeze(), tech_x.squeeze()

        # N, C
        if main_x.dim() == 1:
            main_x, mate_x, tech_x = main_x.unsqueeze(0), mate_x.unsqueeze(0), tech_x.unsqueeze(0)

        # N, C, models
        fusion_x = torch.stack([main_x, mate_x, tech_x], dim=2)

        # N, 1, C, models
        fusion_x = fusion_x.unsqueeze(1)

        x = self.conv1(fusion_x)
        x = self.relu(x)
        x= self.conv2(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc3(x)

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
            params.requires_grad = True


if __name__ == '__main__':
    inputs = torch.rand(4, 3, 224, 224)

    model = FcFusionNet_v1(
        arch='efficientnet_b0',
        #mate_out_num=6,
        #tech_out_num=3,
        #simsiam_weight_path='logs_simsiam/exp02-0710-182433/300_ckpt.tar',
        #mate_weight_path='logs_material/exp02-0712-000307/cv_0/best_model.pth',
        #tech_weight_path='logs_technique/exp02-0712-025133/cv_0/best_model.pth',
        freeze=True
    )
    
    print(model)

    outputs = model(inputs)

    print(outputs.shape)

    model.unfreeze()