import torch
from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):

    def __init__(self, *shapes, num_samples=10000):
        """
        Generate a dummy dataset

        Args:
            *shapes: list of shapes
            num_samples: how many samples to use in this dataset

        Example::

            from pl_bolts.datamodules import DummyDataset

            # mnist dims
            >>> ds = DummyDataset((1, 28, 28), (1,))
            >>> dl = DataLoader(ds, batch_size=7)
            ...
            >>> batch = next(iter(dl))
            >>> x, y = batch
            >>> x.size()
            torch.Size([7, 1, 28, 28])
            >>> y.size()
            torch.Size([7, 1])
        """
        super().__init__()
        self.shapes = shapes
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        samples = []
        for shape in self.shapes:
            sample = torch.rand(*shape)
            samples.append(sample)

        return samples
