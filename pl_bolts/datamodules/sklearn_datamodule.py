from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataLoaders
from torch.utils.data import Dataset, DataLoader, random_split
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


# assumes you have a big X and Y and does the train/val/test split for you
class SklearnDataLoaders(BoltDataLoaders):
    def __init__(self, X, y, val_split=0.15, test_split=0.15, X_transform=None, y_transform=None, num_workers=16,
                 shuffle=True, random_seed=0):
        self.X = X
        self.Y = y
        self.val_split = val_split
        self.test_split = test_split
        self.X_transform = X_transform
        self.y_transform = y_transform
        self.num_workers = num_workers

        # Split X, y into train/validation/test
        dataset_size = len(X)
        indices = list(range(dataset_size))
        val_size = int(np.floor(val_split * dataset_size))
        test_size = int(np.floor(test_split * dataset_size))

        self.dataset_train, self.dataset_val, self.dataset_test = random_split((X, y),
                                                                               [dataset_size - val_size - test_size,
                                                                                val_size,
                                                                                test_size])

    def train_dataloader(self, batch_size):
        loader = DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self, batch_size):
        loader = DataLoader(
            self.dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size):
        loader = DataLoader(
            self.dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader