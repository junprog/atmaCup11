import os

from PIL import Image

import torch
from torch._C import dtype
import torch.utils.data as data

def soft_label(label):
    if 1801 <= label:
        soft = label * 0.002631578947 - 2.239473683

    elif 1600 < label and label < 1801:
        soft = label / 100. - 15.51
    
    elif label <= 1600:
        soft = label * 0.003105590062 - 4.472049689

    return soft

class AtmaDataset(data.Dataset):
    """画像とtargetを返すdataset
    """
    def __init__(self, data_dir, img_name_df, target_df, trans, target_scale=None):
        self.data_dir = data_dir

        self.img_name = list(img_name_df)
        self.label = list(target_df.values)

        self.trans = trans
        self.target_scale = target_scale

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_name[idx] + '.jpg')
        img = Image.open(img_path)
        img = self.trans(img)

        tar = self.label[idx]
        if self.target_scale is not None:
            tar = tar * self.target_scale

        tar = torch.Tensor(list(tar))

        return img, tar

class MultiAtmaDataset(data.Dataset):
    """画像とtargetを返すdataset
    """
    def __init__(self, data_dir, img_name_df, target_df, mate_df, tech_df, trans, target_scale=None):
        self.data_dir = data_dir

        self.img_name = list(img_name_df)
        self.label = list(target_df.values)
        self.mate = list(mate_df.values)
        self.tech = list(tech_df.values)

        self.trans = trans
        self.target_scale = target_scale

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_name[idx] + '.jpg')
        img = Image.open(img_path)
        img = self.trans(img)

        tar = self.label[idx]
        mate = self.mate[idx]
        tech = self.tech[idx]

        if self.target_scale is not None:
            tar = tar * self.target_scale

        tar = torch.Tensor(list(tar))
        mate = torch.Tensor(list(mate))
        tech = torch.Tensor(list(tech))

        return img, tar, mate, tech
