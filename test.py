import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.models as models

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
pl.seed_everything(765)

DATA_DIR = 'C:/Users/Junya/Desktop/dataset_atmaCup11'

train_csv_path = os.path.join(DATA_DIR, 'train.csv')
test_csv_path = os.path.join(DATA_DIR, 'test.csv')
material_path = os.path.join(DATA_DIR, 'material.csv')
techniques_path = os.path.join(DATA_DIR, 'techniques.csv')

test_df = pd.read_csv(test_csv_path)

class ResNet18(pl.LightningModule):
    def __init__(self, out_dim=1):
        super().__init__()

        resnet18 = models.resnet18(pretrained=False)
        layers = list(resnet18.children())[:-1]

        self.feature = nn.Sequential(*layers)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.feature(x)
        x = x.squeeze()
        out = self.fc(x)

        return out

### training roop
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

submission = pd.DataFrame()

#res = np.zeros((len(test_df), 1), dtype=np.float32)
res = np.zeros((6, 1), dtype=np.float32)
for model_path in ['lightning_logs/exp01/version_0/checkpoints/epoch=44-step=2204.ckpt', 'lightning_logs/exp01/version_1/checkpoints/epoch=91-step=4507.ckpt', 'lightning_logs/exp01/version_2/checkpoints/epoch=67-step=3331.ckpt', 'lightning_logs/exp01/version_3/checkpoints/epoch=84-step=4164.ckpt', 'lightning_logs/exp01/version_4/checkpoints/epoch=94-step=4654.ckpt']:
    cv_res = []
    model = ResNet18().load_from_checkpoint(model_path).eval().cuda()
    for i, test in enumerate(tqdm(test_df['object_id'])):
        img_path = os.path.join(DATA_DIR, 'photos', test + '.jpg')
        img = Image.open(img_path)
        img = val_transforms(img)
        img.unsqueeze_(0)

        img = img.cuda()

        output = model(img).data.cpu().numpy()
        cv_res.append(output)

    res += np.array(cv_res)

submission['target'] = list(np.squeeze(res / 5))
submission['target'].to_csv('submission.csv', index=False)