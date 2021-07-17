import torch.nn as nn

from models.efficient_net import EfficientNet
from models.resnet import ResNet
from models.vision_transformer import ViT


class projection_MLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=512):
        super().__init__()
    
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        if x.dim() != 2:
            x = x.squeeze()
        x = self.layers(x)
        return x 

class prediction_MLP(nn.Module):
    def __init__(self, bn=True, in_dim=512, hidden_dim=256, out_dim=512):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        if x.dim() != 2:
            x = x.squeeze()
        x = self.layer1(x)
        x = self.layer2(x)

class SiamNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        if 'resnet' in arch:
            m = ResNet(arch, 1)
        elif 'efficientnet' in arch:
            m = EfficientNet(arch, 1)
        elif 'vit' in arch:
            m = ViT(num_classes=1)

        self.encoder = nn.Sequential(*list(m.children())[:-1])

        if 'resnet' in arch:
            self.projector = projection_MLP(in_dim=512, hidden_dim=512, out_dim=512)
            self.predictor = prediction_MLP(in_dim=512, hidden_dim=256, out_dim=512)
        elif 'efficientnet' in arch:
            self.projector = projection_MLP(in_dim=1280, hidden_dim=512, out_dim=512)
            self.predictor = prediction_MLP(in_dim=512, hidden_dim=256, out_dim=512)
        elif 'vit' in arch:
            self.projector = projection_MLP(in_dim=384, hidden_dim=512, out_dim=512)
            self.predictor = prediction_MLP(in_dim=512, hidden_dim=256, out_dim=512)

    def forward(self, input1, input2, test=False):
        if test:
            f = self.encoder
            z = f(input1)

            return z
            
        else:
            f, h = self.encoder, self.predictor

            z1, z2 = f(input1), f(input2)

            z1, z2 = self.projector(z1), self.projector(z2)
            p1, p2 = h(z1), h(z2)

            return (z1, z2), (p1, p2)

        
