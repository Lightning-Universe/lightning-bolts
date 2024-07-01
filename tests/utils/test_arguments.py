from dataclasses import FrozenInstanceError

import pytest
from lightning.pytorch import LightningDataModule, LightningModule
from pl_bolts.utils.arguments import LightningArgumentParser, LitArg, gather_lit_args


class DummyParentModel(LightningModule):
    name = "parent-model"

    def __init__(self, a: int, b: str, c: str = "parent_model_c") -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        pass


class DummyParentDataModule(LightningDataModule):
    name = "parent-dm"

    def __init__(self, d: str, c: str = "parent_dm_c") -> None:
        super().__init__()
        self.d = d
        self.c = c


def test_lightning_argument_parser():
    parser = LightningArgumentParser(ignore_required_init_args=False)
    assert parser.ignore_required_init_args is False
    parser = LightningArgumentParser(ignore_required_init_args=True)
    assert parser.ignore_required_init_args is True


@pytest.mark.xfail()
def test_parser_bad_argument():
    parser = LightningArgumentParser()
    parser.add_object_args("dm", DummyParentDataModule)
    parser.add_object_args("model", DummyParentModel)
    parser.parse_lit_args(["--some-bad-arg", "asdf"])


def test_lit_arg_immutable():
    arg = LitArg("some_arg", (int,), 1, False)
    with pytest.raises(FrozenInstanceError):
        arg.default = 0
    assert arg.default == 1


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        pytest.param(
            DummyParentModel,
            {
                "a": (int, None, True),
                "b": (str, None, True),
                "c": (str, "parent_model_c", False),
            },
            id="dummy-parent-model",
        ),
        pytest.param(
            DummyParentDataModule,
            {
                "d": (str, None, True),
                "c": (str, "parent_dm_c", False),
            },
            id="dummy-parent-dm",
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
    ("ignore_required_init_args", "dm_cls", "model_cls", "a", "b", "c", "d"),
    [
        pytest.param(True, DummyParentDataModule, DummyParentModel, 999, "bbb", "ccc", "ddd", id="base"),
    ],
)
def test_lightning_arguments(ignore_required_init_args, dm_cls, model_cls, a, b, c, d):
    parser = LightningArgumentParser(ignore_required_init_args=ignore_required_init_args)
    parser.add_object_args("dm", dm_cls)
    parser.add_object_args("model", model_cls)

    mocked_args = f"""
        --a 1
        --b {b}
        --c {c}
        --d {d}
    """.strip().split()
    args = parser.parse_lit_args(mocked_args)
    assert vars(args.dm)["c"] == vars(args.model)["c"] == c
    assert "d" not in args.dm
