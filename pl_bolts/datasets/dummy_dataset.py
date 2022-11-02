import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """Generate a dummy dataset.

    Example:
        >>> from pl_bolts.datasets import DummyDataset
        >>> from torch.utils.data import DataLoader
        >>> # mnist dims
        >>> ds = DummyDataset((1, 28, 28), (1, ))
        >>> dl = DataLoader(ds, batch_size=7)
        >>> # get first batch
        >>> batch = next(iter(dl))
        >>> x, y = batch
        >>> x.size()
        torch.Size([7, 1, 28, 28])
        >>> y.size()
        torch.Size([7, 1])
    """

    def __init__(self, *shapes, num_samples: int = 10000):
        """
        Args:
            *shapes: list of shapes
            num_samples: how many samples to use in this dataset
        """
        super().__init__()
        self.shapes = shapes

        if num_samples < 1:
            raise ValueError("Provide an argument greater than 0 for `num_samples`")

        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        sample = []
        for shape in self.shapes:
            spl = torch.rand(*shape)
            sample.append(spl)
        return sample


class DummyDetectionDataset(Dataset):
    """Generate a dummy dataset for object detection.

    Example:
        >>> from pl_bolts.datasets import DummyDetectionDataset
        >>> from torch.utils.data import DataLoader
        >>> ds = DummyDetectionDataset()
        >>> dl = DataLoader(ds, batch_size=7)
        >>> # get first batch
        >>> batch = next(iter(dl))
        >>> x,y = batch
        >>> x.size()
        torch.Size([7, 3, 256, 256])
        >>> y['boxes'].size()
        torch.Size([7, 1, 4])
        >>> y['labels'].size()
        torch.Size([7, 1])
    """

    def __init__(
        self, img_shape: tuple = (3, 256, 256), num_boxes: int = 1, num_classes: int = 2, num_samples: int = 10000
    ):
        """
        Args:
            *shapes: list of shapes
            num_samples: how many samples to use in this dataset
        """
        super().__init__()
        if num_samples < 1:
            raise ValueError("Provide an argument greater than 0 for `num_samples`")

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

    def __getitem__(self, idx: int):
        img = torch.rand(self.img_shape)
        boxes = torch.tensor([self._random_bbox() for _ in range(self.num_boxes)], dtype=torch.float32)
        labels = torch.randint(self.num_classes, (self.num_boxes,), dtype=torch.long)
        return img, {"boxes": boxes, "labels": labels}


class RandomDictDataset(Dataset):
    """Generate a dummy dataset with a dict structure.

    Example:
        >>> from pl_bolts.datasets import RandomDictDataset
        >>> from torch.utils.data import DataLoader
        >>> ds = RandomDictDataset(10)
        >>> dl = DataLoader(ds, batch_size=7)
        >>> batch = next(iter(dl))
        >>> len(batch['a']),len(batch['a'][0])
        (7, 10)
        >>> len(batch['b']),len(batch['b'][0])
        (7, 10)
    """

    def __init__(self, size: int, num_samples: int = 250):
        """
        Args:
            size: integer representing the length of a feature_vector
            num_samples: number of samples
        """
        if num_samples < 1:
            raise ValueError("Provide an argument greater than 0 for `num_samples`")

        if size < 1:
            raise ValueError("Provide an argument greater than 0 for `size`")

        self.len = num_samples
        self.data_a = torch.randn(num_samples, size)
        self.data_b = torch.randn(num_samples, size)

    def __getitem__(self, index):
        a = self.data_a[index]
        b = self.data_b[index]
        return {"a": a, "b": b}

    def __len__(self):
        return self.len


class RandomDictStringDataset(Dataset):
    """Generate a dummy dataset with in dict structure with strings as indexes.

    Example:
        >>> from pl_bolts.datasets import RandomDictStringDataset
        >>> from torch.utils.data import DataLoader
        >>> ds = RandomDictStringDataset(10)
        >>> dl = DataLoader(ds, batch_size=7)
        >>> batch = next(iter(dl))
        >>> batch['id']
        ['0', '1', '2', '3', '4', '5', '6']
        >>> len(batch['x'])
        7
    """

    def __init__(self, size: int, num_samples: int = 250):
        """
        Args:
            size: tuple
            num_samples: number of samples
        """
        if num_samples < 1:
            raise ValueError("Provide an argument greater than 0 for `num_samples`")

        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        return {"id": str(index), "x": self.data[index]}

    def __len__(self):
        return self.len


class RandomDataset(Dataset):
    """Generate a dummy dataset.

    Example:
        >>> from pl_bolts.datasets import RandomDataset
        >>> from torch.utils.data import DataLoader
        >>> ds = RandomDataset(10)
        >>> dl = DataLoader(ds, batch_size=7)
        >>> batch = next(iter(dl))
        >>> len(batch),len(batch[0])
        (7, 10)
    """

    def __init__(self, size: int, num_samples: int = 250):
        """
        Args:
            size: tuple
            num_samples: number of samples
        """
        if num_samples < 1:
            raise ValueError("Provide an argument greater than 0 for `num_samples`")

        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
