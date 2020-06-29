import os
from pl_bolts.datamodules.dac.dac_preprocess import preprocess_dac
from torch.utils.data import DataLoader, random_split
from pl_bolts.datamodules.lightning_datamodule import LightningDataModule
from torch.utils.data import Dataset
import numpy as np
import torch

try:
    import pandas as pd
except Exception as e:
    raise Exception('To use this datamodule please install pandas\npip install pandas')

URL = 'https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz'
URL_preprocessed = 'https://pl-public-data.s3.amazonaws.com/doc_processed.zip'
URL_preprocessed_tiny = 'https://pl-public-data.s3.amazonaws.com/doc_processed_tiny.zip'


class DACDataModule(LightningDataModule):

    name = 'kadac'

    def __init__(
            self,
            data_dir: str,
            val_split: float = 0.15,
            num_workers: int = 16,
            continuous_features: int = 13,
            download_preprocessed: bool = True,
            use_tiny_dac: bool = False,
            *args,
            **kwargs,
    ):
        """
        Dataset from `Kaggle Display Advertising Challenge
        <https://www.kaggle.com/c/criteo-display-ad-challenge>`_.

        Example::

            from pl_bolts.datamodules import DACDataModule
            from pl_bolts.models.recommenders import DLRM

            dm = DACDataModule('.')
            model = DLRM(datamodule=dm)

        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            continuous_features: num of continuous features to use
            download_preprocessed: if true, downloads preprocessed. Otherwise it will download and preprocess
                (use in case you want to verify the preprocessing).
            use_tiny_dac: if true, uses a smaller version with only 100_000 samples
        """
        super().__init__(*args, **kwargs)
        self.dims = (1, 28, 28)
        self.data_dir = os.path.join(data_dir, 'dac')
        self.val_split = val_split
        self.num_workers = num_workers
        self.continuous_features = continuous_features
        self.download_preprocessed = download_preprocessed
        self.use_tiny_dac = use_tiny_dac

        # this dataset needs to be downloaded from here and untared (this module automates it)
        url = URL
        if download_preprocessed and use_tiny_dac:
            url = URL_preprocessed_tiny
        elif download_preprocessed:
            url = URL_preprocessed
        self.add_data_url(url)

    @property
    def num_classes(self):
        """
        Return:
            10
        """
        return 10

    def prepare_data(self):
        """
        Download and untar the dataset

        Deletes the original dataset and only keeps the untarred files
        """
        preprocess_dac(self.data_dir, self.data_dir, num_train_sample=100_000, num_test_sample=100_000)

        # num_downloaded_files = self.download_registered_data_urls()
        # if num_downloaded_files > 0 and not self.download_preprocessed:
        #     num_samples = 100_000 if self.use_tiny_dac else -1
        #     preprocess_dac(self.data_dir, self.data_dir, num_train_sample=num_samples, num_test_sample=num_samples)

    def train_dataloader(self, batch_size=32):
        """
        DAC train dataset (`train.txt` file). Some part is held out for validation

        Args:
            batch_size: size of batch
        """

        train_split, _ = self.__train_val_split()

        loader = DataLoader(
            train_split,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def __train_val_split(self):
        # get the train data
        root = os.path.join(self.data_dir, 'train_processed.txt')
        data = pd.read_csv(root)
        train_data = data.iloc[:, :-1].values
        target = data.iloc[:, -1].values

        num_val = int(len(train_data) * self.val_split)
        num_train = len(train_data) - num_val
        dataset = DACDataset(train_data, target)
        train_split, val_split = random_split(dataset, [num_train, num_val])

        return train_split, val_split

    def val_dataloader(self, batch_size=32):
        """
        DAC train dataset (`train.txt` file). The part held out for validation

        Args:
            batch_size: size of batch
        """
        _, val_split = self.__train_val_split()

        loader = DataLoader(
            val_split,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size=32):
        """
        DAC test dataset (`test.txt` file). The part held out for validation

        Args:
            batch_size: size of batch
        """
        root = os.path.join(self.data_dir, 'test_processed.txt')
        data = pd.read_csv(root)
        test_data = data.iloc[:, :-1].values

        dataset = DACDataset(test_data, test=True)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader


class DACDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray = None, test: bool = False, num_continuous_feats: int = 13):
        super().__init__()

        self.X = X
        self.y = y
        self.test = test
        self.num_continuous_feats = num_continuous_feats

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.test:
            dp = self.X.iloc[idx, :]

            # get continuous features (idx=1)
            x_continuous = np.ones_like(dp[:self.num_continuous_feats])
            x_categorical = dp[self.num_continuous_feats:]
            x1 = torch.from_numpy(np.concatenate((x_continuous, x_categorical)).astype(np.int32)).unsqueeze(-1)

            # one hot features (value = 1)
            x2_categorical = np.ones_like(dp[self.num_continuous_feats:])
            x2_continuous = dp[:self.num_continuous_feats]
            x2 = torch.from_numpy(np.concatenate((x2_continuous, x2_categorical)).astype(np.int32))
            return x1, x2
        else:
            dp, y = self.X[idx, :], self.y[idx]

            # get continuous features (idx=0)
            x_continuous = np.zeros_like(dp[:self.num_continuous_feats])
            x_categorical = dp[self.num_continuous_feats:]
            x1 = torch.from_numpy(np.concatenate((x_continuous, x_categorical)).astype(np.int32)).unsqueeze(-1)

            # one hot features (value = 1)
            x2_categorical = np.ones_like(dp[self.num_continuous_feats:])
            x2_continuous = dp[:self.num_continuous_feats]
            x2 = torch.from_numpy(np.concatenate((x2_continuous, x2_categorical)).astype(np.int32))
            return x1, x2, y


if __name__ == '__main__':
    dm = DACDataModule(data_dir='/Users/williamfalcon/Downloads/', use_tiny_dac=True, download_preprocessed=False)
    dm.prepare_data()

    for batch in dm.train_dataloader():
        (x1, x2, y) = batch
        print(batch)
        break

    for batch in dm.val_dataloader():
        (x1, x2) = batch
        print(batch)
        break