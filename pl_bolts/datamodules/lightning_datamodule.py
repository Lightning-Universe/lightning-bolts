import inspect
from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Union, List, Tuple, Any

from pytorch_lightning.utilities import rank_zero_warn, parsing
from torch.utils.data import DataLoader


class LightningDataModule(object):
    """
    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):

            def __init__(self):
                super().__init__()

            def prepare_data(self):
                # download, split, etc...

            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)

            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)

            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)

    A DataModule implements 4 key methods

    1. **prepare_data** (things to do on 1 GPU not on every GPU in distributed mode)
    2. **train_dataloader** the training dataloader.
    3. **val_dataloader** the val dataloader.
    4. **test_dataloader** the test dataloader.


    This allows you to share a full dataset without explaining what the splits, transforms or download
    process is.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def size(self):
        """
        Return the dimension of each input
        Either as a tuple or list of tuples
        """
        raise NotImplementedError

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

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        r"""Extends existing argparse by default `LightningDataModule` attributes.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False,)
        added_args = [x.dest for x in parser._actions]

        blacklist = ['kwargs']
        depr_arg_names = blacklist + added_args
        depr_arg_names = set(depr_arg_names)

        allowed_types = (str, float, int, bool)

        # TODO: get "help" from docstring :)
        for arg, arg_types, arg_default in (at for at in cls.get_init_arguments_and_types()
                                            if at[0] not in depr_arg_names):
            arg_types = [at for at in allowed_types if at in arg_types]
            if not arg_types:
                # skip argument with not supported type
                continue
            arg_kwargs = {}
            if bool in arg_types:
                arg_kwargs.update(nargs="?")
                # if the only arg type is bool
                if len(arg_types) == 1:
                    # redefine the type for ArgParser needed
                    def use_type(x):
                        return bool(parsing.str_to_bool(x))
                else:
                    # filter out the bool as we need to use more general
                    use_type = [at for at in arg_types if at is not bool][0]
            else:
                use_type = arg_types[0]

            if arg_default == inspect._empty:
                arg_default = None

            parser.add_argument(
                f'--{arg}',
                dest=arg,
                default=arg_default,
                type=use_type,
                help=f'autogenerated by plb.{cls.__name__}',
                **arg_kwargs,
            )

        return parser

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
        """
        Create an instance from CLI arguments.

        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
                parsed and passed to the :class:`LightningDataModule`.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
                These must be valid Trainer arguments.

        Example::

            parser = ArgumentParser(add_help=False)
            parser = LightningDataModule.add_argparse_args(parser)
            module = LightningDataModule.from_argparse_args(args)
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid Trainer args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        trainer_kwargs = dict((name, params[name]) for name in valid_kwargs if name in params)
        trainer_kwargs.update(**kwargs)

        return cls(**trainer_kwargs)

    @classmethod
    def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
        r"""Scans the Trainer signature and returns argument names, types and default values.

        Returns:
            List with tuples of 3 values:
            (argument name, set with argument types, argument default value).
        """
        trainer_default_params = inspect.signature(cls).parameters
        name_type_default = []
        for arg in trainer_default_params:
            arg_type = trainer_default_params[arg].annotation
            arg_default = trainer_default_params[arg].default
            try:
                arg_types = tuple(arg_type.__args__)
            except AttributeError:
                arg_types = (arg_type,)

            name_type_default.append((arg, arg_types, arg_default))

        return name_type_default
