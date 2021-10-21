import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeDataset(Dataset):

    def __init__(self, num_obj, num_imgs, img_width=8, img_height=8, device="cpu"):
        imgs = np.zeros((num_imgs, img_width, img_height))
        bbox = np.zeros((num_imgs, num_obj, 4))

        for i in range(num_imgs):
            for o in range(num_obj):
                w = np.random.randint(1, img_width)
                h = np.random.randint(1, img_height)
                x = np.random.randint(0, img_width - w)
                y = np.random.randint(0, img_height - h)
                imgs[i, x:x + w, y:y + h] = 1
                bbox[i, o] = [x, y, w, h]

        self._len = len(imgs)
        self._imgs = torch.tensor(imgs).unsqueeze(1).to(device)
        self._bbox = torch.tensor(bbox).to(device)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self._imgs[idx], self._bbox[idx]

    @property
    def data(self):
        return self._imgs

    @property
    def targets(self):
        return self._bbox

    def to_inplace(self, device):
        self._imgs = self._imgs.to(device)
        self._bbox = self._bbox.to(device)
