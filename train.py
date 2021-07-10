import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

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

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

BATCH_SIZE = 64
L_RATE = 1e-2

class atmaDataset(data.Dataset):
    def __init__(self, img_name_df, target_df, trans):
        self.img_name = list(img_name_df)
        self.label = list(target_df)

        self.trans = trans

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_path = os.path.join(DATA_DIR, 'photos', self.img_name[idx] + '.jpg')
        img = Image.open(img_path)
        img = self.trans(img)

        tar = self.label[idx]
        tar = torch.tensor(tar, dtype=torch.float32).unsqueeze(-1)

        return img, tar

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

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        criterion = nn.MSELoss()
        loss = criterion(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        criterion = nn.MSELoss()
        loss = criterion(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=L_RATE)
        return optimizer

### training roop
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for train, val in skf.split(train_df['object_id'], train_df['target']):
    tr_dataset = atmaDataset(train_df.object_id[train], train_df.target[train], train_transforms)
    vl_dataset = atmaDataset(train_df.object_id[val], train_df.target[val], val_transforms)

    train_loader = torch.utils.data.DataLoader(tr_dataset, BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(vl_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

    net = ResNet18()
    #logger = CSVLogger(save_dir='logs', name='my_exp')
    trainer = pl.Trainer(max_epochs=100, gpus=1, deterministic=True)#, logger=logger)
    trainer.fit(net, train_loader, val_loader)