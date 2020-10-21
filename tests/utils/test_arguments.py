from dataclasses import FrozenInstanceError

import pytest
import pytorch_lightning as pl

from pl_bolts.utils.arguments import LightningArgumentParser, LitArg, gather_lit_args


class DummyBaseModel(pl.LightningModule):

    name = "base-model"

    def __init__(self, input_dim: int, hidden_dim: int = 64, batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()


class DummyChildModel(DummyBaseModel):

    name = "child-model"

    def __init__(self, *args, num_classes: int, freeze_encoder: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()


class DummyBaseDataModule(pl.LightningDataModule):

    name = "base-dm"

    def __init__(self, root: str, batch_size: int = 16, num_workers: int = 8):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers


def test_lit_argument_parser_required_init_args():
    parser = LightningArgumentParser(ignore_required_init_args=False)
    assert parser.ignore_required_init_args is False
    parser = LightningArgumentParser(ignore_required_init_args=True)
    assert parser.ignore_required_init_args is True


@pytest.mark.xfail()
def test_parser_bad_argument():
    parser = LightningArgumentParser()
    parser.add_datamodule_args(DummyBaseDataModule)
    parser.add_model_args(DummyBaseModel)
    args = parser.parse_lit_args(['--some-bad-arg', 'asdf'])


def test_lit_arg_immutable():
    arg = LitArg("some_arg", (int,), 1, False)
    with pytest.raises(FrozenInstanceError):
        arg.default = 0
    assert arg.default == 1


@pytest.mark.parametrize(
    "obj,expected",
    [
        pytest.param(
            DummyBaseModel,
            {
                "input_dim": (int, None, True),  # Tuples here are (expected type, default, is_required)
                "hidden_dim": (int, 64, False),
                "batch_size": (int, 32, False),
            },
            id="dummy-base-model",
        ),
        pytest.param(
            DummyBaseDataModule,
            {
                "root": (str, None, True),
                "batch_size": (int, 16, False),
                "num_workers": (int, 8, False)
            },
            id="dummy-base-dm",
        ),
        pytest.param(
            DummyChildModel,
            {   
                "num_classes": (int, None, True),
                "freeze_encoder": (bool, False, False),
                "input_dim": (int, None, True),
                "hidden_dim": (int, 64, False),
                "batch_size": (int, 32, False),
            },
            id="dummy-child-model",
        ),
    ],
)
def test_gather_lit_args(obj, expected):
    lit_args = gather_lit_args(obj)
    assert len(lit_args) == len(expected)
    for lit_arg, (k, v) in zip(lit_args, expected.items()):
        assert lit_arg.name == k
        assert lit_arg.types[0] == v[0]
        assert lit_arg.default == v[1]
        assert lit_arg.required == v[2]


@pytest.mark.parametrize(
    "dm_cls,model_cls,mocked_args,expected_dm_args,expected_model_args,expected_trainer_args",
    [
        pytest.param(
            DummyBaseDataModule,
            DummyBaseModel,
            '',
            {'batch_size': 16, 'num_workers': 8},
            {'batch_size': 16, 'hidden_dim': 64},
            {},
            id="base",
        ),
        pytest.param(
            DummyBaseDataModule,
            DummyChildModel,
            '',
            {'batch_size': 16, 'num_workers': 8},
            {'batch_size': 16, 'hidden_dim': 64, 'freeze_encoder': False},
            {},
            id="child-model",
        ),
    ]
)
def test_base_usage(dm_cls, model_cls, mocked_args, expected_dm_args, expected_model_args, expected_trainer_args):
    parser = LightningArgumentParser()
    parser.add_datamodule_args(dm_cls)
    parser.add_model_args(model_cls)
    parser.add_trainer_args()
    args = parser.parse_lit_args('')

    for arg_name, expected_value in expected_dm_args.items():
        assert getattr(args.datamodule, arg_name, None) == expected_value

    for arg_name, expected_value in expected_model_args.items():
        assert getattr(args.model, arg_name, None) == expected_value

    for arg_name, expected_value in expected_trainer_args.items():
        assert getattr(args.trainer, arg_name, None) == expected_value
