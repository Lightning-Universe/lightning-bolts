from abc import abstractmethod
from typing import List, Union
import inspect

from pytorch_lightning.utilities import rank_zero_warn
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace


class BoltDataModule(object):
    def __init__(self):
        """
        A DataModule standardizes that training, val, test splits, data preparation and transforms.
        The main advantage is consistent data splits across models.

        A DataModule implements 4 key methods

        The first is `prepare_data`. In Lightning, this method ensures that data processing and downloading
        only happens on 1 GPU when doing multi-gpu training.

        Prepare Data::

            def prepare_data(self):
                # download, split, etc...

        The other three methods generate dataloaders for each split of the dataset. Each dataloader is also optimized
        with best practices of num_workers, pinning, etc.

        Dataloaders::

            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)

            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)

            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)

        Another key problem DataModules solve is that it standardizes transforms (ie: no need to figure out the
        normalization coefficients for a dataset).
        """
        super().__init__()

    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        """
        Use this to download and prepare data.
        In distributed (GPU, TPU), this will only be called once.
        This is called before requesting the dataloaders:

        .. code-block:: python

            model.prepare_data()
            model.train_dataloader()
            model.val_dataloader()
            model.test_dataloader()

        Examples:
            .. code-block:: python

                def prepare_data(self):
                    download_imagenet()
                    clean_imagenet()
                    cache_imagenet()
        """

    @abstractmethod
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        Implement a PyTorch DataLoader for training.

        Return:
            Single PyTorch :class:`~torch.utils.data.DataLoader`.

        The dataloader you return will not be called every epoch unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to ``True``.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~pytorch_lightning.trainer.Trainer.fit`
        - ...
        - :meth:`prepare_data`
        - :meth:`train_dataloader`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Example:
            .. code-block:: python

                def train_dataloader(self):
                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (1.0,))])
                    dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform,
                                    download=True)
                    loader = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=self.hparams.batch_size,
                        shuffle=True
                    )
                    return loader

        """
        rank_zero_warn('`train_dataloader` must be implemented to be used with the Lightning Trainer')

    @abstractmethod
    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        r"""
        Implement one or multiple PyTorch DataLoaders for validation.

        The dataloader you return will not be called every epoch unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to ``True``.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~pytorch_lightning.trainer.Trainer.fit`
        - ...
        - :meth:`prepare_data`
        - :meth:`train_dataloader`
        - :meth:`val_dataloader`
        - :meth:`test_dataloader`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Return:
            Single or multiple PyTorch DataLoaders.

        Examples:
            .. code-block:: python

                def val_dataloader(self):
                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (1.0,))])
                    dataset = MNIST(root='/path/to/mnist/', train=False,
                                    transform=transform, download=True)
                    loader = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=self.hparams.batch_size,
                        shuffle=True
                    )

                    return loader

                # can also return multiple dataloaders
                def val_dataloader(self):
                    return [loader_a, loader_b, ..., loader_n]

        Note:
            If you don't need a validation dataset and a :meth:`validation_step`, you don't need to
            implement this method.

        Note:
            In the case where you return multiple validation dataloaders, the :meth:`validation_step`
            will have an argument ``dataset_idx`` which matches the order here.
        """

    @abstractmethod
    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        r"""
        Implement one or multiple PyTorch DataLoaders for testing.

        The dataloader you return will not be called every epoch unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to ``True``.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~pytorch_lightning.trainer.Trainer.fit`
        - ...
        - :meth:`prepare_data`
        - :meth:`train_dataloader`
        - :meth:`val_dataloader`
        - :meth:`test_dataloader`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Return:
            Single or multiple PyTorch DataLoaders.

        Example:
            .. code-block:: python

                def test_dataloader(self):
                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (1.0,))])
                    dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform,
                                    download=True)
                    loader = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=self.hparams.batch_size,
                        shuffle=True
                    )

                    return loader

        Note:
            If you don't need a test dataset and a :meth:`test_step`, you don't need to implement
            this method.

        """

    @property
    def add_argparse_args(self, parent_parser):
        """
        Adds dataset arguments to an argparser

        Example::

            datamodule = DataModule()
            datamodule = datamodule.add_argparse_args
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str, default='')
        return parser

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs) -> `BoltDataModule`:
        """
        Create an instance from CLI arguments.

        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
                parsed and passed to the :class:`Trainer`.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
                These must be valid Trainer arguments.

        Example::

            parser = ArgumentParser(add_help=False)
            parser = BoltDataModule.add_argparse_args(parser)
            module = BoltDataModule.from_argparse_args(args)
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid Trainer args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        trainer_kwargs = dict((name, params[name]) for name in valid_kwargs if name in params)
        trainer_kwargs.update(**kwargs)

        return cls(**trainer_kwargs)
