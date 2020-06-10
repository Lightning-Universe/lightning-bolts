from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataLoaders
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class SklearnDataset(Dataset):
    def __init__(self, X, y, X_transform=None, y_transform=None):
        self.X = X
        self.Y = y
        self.X_transform = X_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.X_transform:
            x = self.X_transform(x)

        if self.y_transform:
            y = self.y_transform(y)

        return x, y



class SklearnDataLoaders(BoltDataLoaders):
    def __init__(self, X, y, val_split=0.15, test_split=0.15, X_transform=None, y_transform=None):
        self.X = X
        self.Y = y
        self.val_split = val_split
        self.test_split = test_split
        self.X_transform = X_transform
        self.y_transform = y_transform




    def train_dataloader(self):

    def val_dataloader(self):

    def test_dataloader(self):


