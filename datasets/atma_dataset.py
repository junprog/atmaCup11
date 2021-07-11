import os

from PIL import Image

import torch
import torch.utils.data as data

class AtmaDataset(data.Dataset):
    """画像とtargetを返すdataset
    """
    def __init__(self, data_dir, img_name_df, target_df, trans):
        self.data_dir = data_dir

        self.img_name = list(img_name_df)
        self.label = list(target_df.values)

        self.trans = trans

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_name[idx] + '.jpg')
        img = Image.open(img_path)
        img = self.trans(img)

        tar = self.label[idx]
        tar = torch.Tensor(list(tar))

        return img, tar
