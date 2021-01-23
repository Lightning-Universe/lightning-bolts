import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch import Tensor, LongTensor
from torch.utils.data import TensorDataset, DataLoader

from pl_bolts.models.diffeq.neuralode import DepthInvariantNeuralODE
from torchdyn.datasets import ToyDataset

def test_neural_ode(tmpdir):
    seed_everything()
    dset = ToyDataset()
    X, yn = dset.generate(n_samples=512, dataset_type='moons', noise=.4)
    X_train, y_train = Tensor(X), LongTensor(yn.long())
    train = TensorDataset(X_train, y_train)
    loader = DataLoader(train, batch_size=len(X), shuffle=False)

    model = DepthInvariantNeuralODE(input_dim=2, hidden_dim=32)
    trainer = pl.Trainer(max_epochs=300, default_root_dir=tmpdir, progress_bar_refresh_rate=0)
    trainer.fit(model, loader, loader)
    assert trainer.logged_metrics['loss'] < 1e-1

