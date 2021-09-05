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
import os
from pathlib import Path

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from pl_bolts.callbacks import SparseMLCallback
from pl_bolts.utils import _SPARSEML_AVAILABLE
from tests.helpers.boring_model import BoringModel

if _SPARSEML_AVAILABLE:
    from sparseml.pytorch.optim import RecipeManagerStepWrapper


@pytest.fixture
def recipe():
    return """
    version: 0.1.0
    modifiers:
    - !EpochRangeModifier
        start_epoch: 0.0
        end_epoch: 1.0

    - !LearningRateModifier
        start_epoch: 0
        end_epoch: -1.0
        update_frequency: -1.0
        init_lr: 0.005
        lr_class: MultiStepLR
        lr_kwargs: {'milestones': [43, 60], 'gamma': 0.1}

    - !GMPruningModifier
        start_epoch: 0
        end_epoch: 40
        update_frequency: 1.0
        init_sparsity: 0.05
        final_sparsity: 0.85
        mask_type: unstructured
        params: __ALL__
    """


@pytest.mark.skipif(not _SPARSEML_AVAILABLE, reason="SparseML isn't installed.")
def test_train_sparse_ml_callback(tmpdir, recipe):
    class TestCallback(Callback):
        def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
            assert isinstance(trainer.optimizers[0], RecipeManagerStepWrapper)

    recipe_path = Path(tmpdir) / "recipe.yaml"
    recipe_path.write_text(recipe)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        callbacks=[SparseMLCallback(recipe_path=str(recipe_path)), TestCallback()],
    )
    trainer.fit(model)

    sample_batch = torch.randn(1, 32)
    output_dir = Path(tmpdir) / "model_export/"
    SparseMLCallback.export_to_sparse_onnx(model, output_dir, sample_batch=sample_batch)
    assert os.path.exists(output_dir)


@pytest.mark.skipif(not _SPARSEML_AVAILABLE, reason="SparseML isn't installed.")
def test_fail_if_no_example_input_array_or_sample_batch(tmpdir, recipe):
    model = BoringModel()
    with pytest.raises(MisconfigurationException, match="To export the model, a sample batch must be passed"):
        output_dir = Path(tmpdir) / "model_export/"
        SparseMLCallback.export_to_sparse_onnx(model, output_dir)


@pytest.mark.skipif(not _SPARSEML_AVAILABLE, reason="SparseML isn't installed.")
def test_fail_if_multiple_optimizers(tmpdir, recipe):
    recipe_path = Path(tmpdir) / "recipe.yaml"
    recipe_path.write_text(recipe)

    class TestModel(BoringModel):
        def configure_optimizers(self):
            return [torch.optim.Adam(self.parameters()), torch.optim.Adam(self.parameters())], []

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, callbacks=[SparseMLCallback(recipe_path=str(recipe_path))]
    )
    with pytest.raises(MisconfigurationException, match="SparseML only supports training with one optimizer."):
        trainer.fit(model)
