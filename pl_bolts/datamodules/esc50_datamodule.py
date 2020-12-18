import os
import numpy as np
import pandas as pd
import requests
import zipfile

from tqdm import tqdm
from pathlib import Path

from typing import Optional

from pl_bolts.datasets.esc50_dataset import ESC50Dataset
from pl_bolts.transforms.audio.wav_to_spectrogram import spec_to_image, get_melspectrogram_db

from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

DATASETS_PATH = '.'

class ESC50DataModule(LightningDataModule):
    """
    Standard ESC50, train, val, test splits and transforms
    """

    name = "esc50"

    def __init__(
        self,
        data_dir: str = DATASETS_PATH,
        num_workers: int = 6,
        seed: int = 42,
        batch_size: int = 16,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            val_size: how many of the val samples to split
            test_size: how many test samples to split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
        """
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.seed = seed
        self.batch_size = batch_size
        self.audio_dir = Path(os.path.join(self.data_dir, 'ESC-50-master/audio'))
        self.spectrograms_dir = Path(os.path.join(self.data_dir, "spectrograms"))
        self.meta_path = Path(os.path.join(self.data_dir, 'ESC-50-master/meta/esc50.csv'))
        self.code_to_label = ...
        self.dataset_train = ...
        self.dataset_val = ...

    @property
    def num_classes(self):
        return 50

    def prepare_data(self):
        """Saves ESC50 files from https://github.com/karolpiczak/ESC-50 to `data_dir`"""
        if self.audio_dir.exists():
            print("ESC50 data already downloaded")
        else:
            zipfile_url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
            response = requests.get(zipfile_url, stream = True)
            # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 1024 #1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open('master.zip', 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            zip_obj = zipfile.ZipFile('master.zip')
            zip_obj.extractall(self.data_dir)

        if self.spectrograms_dir.exists():
            print("ESC50 data is already converted to spectrogram")
        else:
            self.spectrograms_dir.mkdir()
            for wav in tqdm(list(Path(self.audio_dir).glob("*.wav"))):
                spectrogram = spec_to_image(get_melspectrogram_db(wav))
                np.save(self.spectrograms_dir / wav.name, spectrogram)

    def setup(self, stage: Optional[str] = None):
        """Split the train and valid dataset"""
        
        df = pd.read_csv(self.meta_path)
        df["category"] = pd.Categorical(df["category"])
        df["label"] = df["category"].cat.codes
        self.code_to_label = dict(enumerate(df['category'].cat.categories))

        train_df = df[df['fold'] != 5]
        valid_df = df[df['fold'] == 5]
        self.dataset_train = ESC50Data(train_df, dir_path=self.spectrograms_dir)
        self.dataset_val = ESC50Data(valid_df, dir_path=self.spectrograms_dir)


    def train_dataloader(self):
        """ESC50 train set removes a subset to use for validation"""
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """ESC50 val set uses a subset of the training set for validation"""
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

