import os
from PIL import Image

import numpy as np
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class MapDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, trans: transforms = None):
        self.df = pd.read_csv(csv_file, index_col=0)
        self.img_dir = img_dir
        self.transform = trans
        self.num_class = len(pd.unique(self.df['labels']))
        self._filenames = os.listdir(img_dir)
        # self._one_hot_encoding = OneHotEncoder()
        # self._one_hot_encoding.fit(pd.unique(self.df['labels']).reshape(-1, 1))

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.img_dir, self._filenames[item])).convert("RGB")
        label = self.df.loc[self._filenames[item]]['labels']
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.df)





