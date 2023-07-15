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
from typing import Any, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from pl_bolts.utils import _SPARSEML_AVAILABLE, _SPARSEML_TORCH_SATISFIED, _SPARSEML_TORCH_SATISFIED_ERROR

if _SPARSEML_TORCH_SATISFIED:
    from sparseml.pytorch.optim import ScheduledModifierManager
    from sparseml.pytorch.utils import ModuleExporter

from pl_bolts.utils.stability import under_review


@under_review()
class SparseMLCallback(Callback):
    """Enables SparseML aware training. Requires a recipe to run during training.

    Args:
        recipe_path: Path to a SparseML compatible yaml recipe.
            More information at https://docs.neuralmagic.com/sparseml/source/recipes.html

    """

    def __init__(self, recipe_path: str) -> None:
        if not _SPARSEML_AVAILABLE:
            raise MisconfigurationException("SparseML has not be installed, install with pip install sparseml")
        if not _SPARSEML_TORCH_SATISFIED:
            raise MisconfigurationException(_SPARSEML_TORCH_SATISFIED_ERROR)
        self.manager = ScheduledModifierManager.from_yaml(recipe_path)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        optimizer = trainer.optimizers

        if len(optimizer) > 1:
            raise MisconfigurationException("SparseML only supports training with one optimizer.")
        optimizer = optimizer[0]
        optimizer = self.manager.modify(
            pl_module, optimizer, steps_per_epoch=trainer.estimated_stepping_batches, epoch=0
        )
        trainer.optimizers = [optimizer]

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.manager.finalize(pl_module)

    @staticmethod
    def export_to_sparse_onnx(
        model: LightningModule, output_dir: str, sample_batch: Optional[torch.Tensor] = None, **export_kwargs: Any
    ) -> None:
        """Exports the model to ONNX format."""
        with model._prevent_trainer_and_dataloaders_deepcopy():
            exporter = ModuleExporter(model, output_dir=output_dir)
            sample_batch = sample_batch if sample_batch is not None else model.example_input_array  # type: ignore[assignment] # noqa: E501
            if sample_batch is None:
                raise MisconfigurationException(
                    "To export the model, a sample batch must be passed via "
                    "``SparseMLCallback.export_to_sparse_onnx(model, output_dir, sample_batch=sample_batch)`` "
                    "or an ``example_input_array`` property within the LightningModule"
                )
            exporter.export_onnx(sample_batch=sample_batch, **export_kwargs)
