try:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import MNIST
    from PIL import Image
except ImportError:
    raise ImportError('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
                      ' install it with `pip install torchvision`.')


class BinaryMNIST(MNIST):
    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[idx], int(self.targets[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # binary
        img[img < 0.5] = 0.0
        img[img >= 0.5] = 1.0

        return img, target