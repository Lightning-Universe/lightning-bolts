# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from pl_bolts.callbacks import ORTCallback
from pl_bolts.utils import _TORCH_ORT_AVAILABLE
from tests.helpers.boring_model import BoringModel

if _TORCH_ORT_AVAILABLE:
    from torch_ort import ORTModule


@pytest.mark.skipif(not _TORCH_ORT_AVAILABLE, reason="ORT Module aren't installed.")
def test_init_train_enable_ort(tmpdir):
    class TestCallback(Callback):
        def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
            assert isinstance(pl_module.model, ORTModule)

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.model = self.layer

        def forward(self, x):
            return self.model(x)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, callbacks=[ORTCallback(), TestCallback()])
    trainer.fit(model)
    trainer.test(model)


@pytest.mark.skipif(not _TORCH_ORT_AVAILABLE, reason="ORT Module aren't installed.")
def test_ort_callback_fails_no_model(tmpdir):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, callbacks=ORTCallback())
    with pytest.raises(MisconfigurationException, match="Torch ORT requires to wrap a single model"):
        trainer.fit(
            model,
        )
