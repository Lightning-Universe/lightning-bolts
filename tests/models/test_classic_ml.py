import numpy as np
import pytest
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataset
from pl_bolts.models.regression import LinearRegression
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader


@pytest.mark.flaky(reruns=3)
def test_linear_regression_model(tmpdir):
    seed_everything()

    # --------------------
    # numpy data
    # --------------------
    x = np.array([[1.0, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4], [4, 5]])
    y = np.dot(x, np.array([1.0, 2])) + 3
    y = y[:, np.newaxis]
    loader = DataLoader(SklearnDataset(x, y), batch_size=2)

    model = LinearRegression(input_dim=2, learning_rate=0.6)
    trainer = Trainer(
        max_epochs=400,
        default_root_dir=tmpdir,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(
        model,
        loader,
        loader,
    )

    coeffs = model.linear.weight.detach().numpy().flatten()
    np.testing.assert_allclose(coeffs, [1, 2], rtol=1e-3)
    trainer.test(model, loader)
