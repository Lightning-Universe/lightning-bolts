import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

from pl_bolts.transforms.data_normalizations import imagenet_normalization

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg('torchvision')

class ESC50Dataset(Dataset):
    def __init__(self, df, dir_path):
        self.df = df
        self.dir_path = dir_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row["label"]
        data = np.load(f"{self.dir_path}/{row['filename']}.npy").astype(np.float32)
        data = torch.from_numpy(data[np.newaxis,:])
        data = data.repeat(3, 1, 1)  # make spectogram 3 channel
        data = imagenet_normalization(data) # imagenet normalize
        return data, int(label)