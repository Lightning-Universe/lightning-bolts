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

            from pl_bolts.datasets import DummyDataset

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


class DummyDetectionDataset(Dataset):
    def __init__(
        self, img_shape=(3, 256, 256), num_boxes=1, num_classes=2, num_samples=10000
    ):
        """
        Generate a dummy dataset for detection

        Args:
            *shapes: list of shapes
            num_samples: how many samples to use in this dataset

        Example::

            from pl_bolts.datasets import DummyDetectionDataset

            >>> ds = DummyDetectionDataset()
            >>> dl = DataLoader(ds, batch_size=7)
        """
        super().__init__()
        self.img_shape = img_shape
        self.num_samples = num_samples
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def _random_bbox(self):
        c, h, w = self.img_shape
        xs = torch.randint(w, (2,))
        ys = torch.randint(h, (2,))
        return [min(xs), min(ys), max(xs), max(ys)]

    def __getitem__(self, idx):
        img = torch.rand(self.img_shape)
        boxes = torch.tensor([self._random_bbox() for _ in range(self.num_boxes)])
        labels = torch.randint(self.num_classes, (self.num_boxes,))
        return img, {"boxes": boxes, "labels": labels}


class RandomDictDataset(Dataset):
    def __init__(self, size, num_samples):
        """
        Generate a dummy dataset with a dict structure

        Args:
            size: tuple
            num_samples: number of samples

        Example::

            from pl_bolts.datasets import RandomDictDataset

            >>> ds = RandomDictDataset()
            >>> dl = DataLoader(ds, batch_size=7)
        """
        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        a = self.data[index]
        b = a + 2
        return {'a': a, 'b': b}

    def __len__(self):
        return self.len


class RandomDictStringDataset(Dataset):
    def __init__(self, size, num_samples):
        """
        Generate a dummy dataset with strings

        Args:
            size: tuple
            num_samples: number of samples

        Example::

            from pl_bolts.datasets import RandomDictStringDataset

            >>> ds = RandomDictStringDataset()
            >>> dl = DataLoader(ds, batch_size=7)
        """
        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        return {"id": str(index), "x": self.data[index]}

    def __len__(self):
        return self.len


class RandomDataset(Dataset):
    def __init__(self, size, num_samples):
        """
        Generate a dummy dataset

        Args:
            size: tuple
            num_samples: number of samples

        Example::

            from pl_bolts.datasets import RandomDataset

            >>> ds = RandomDataset()
            >>> dl = DataLoader(ds, batch_size=7)
        """
        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
