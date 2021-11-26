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

from pl_bolts.utils import _PL_GREATER_EQUAL_1_4_5, _SPARSEML_AVAILABLE, _TORCH_MAX_VERSION_SPARSEML

if _SPARSEML_AVAILABLE:
    from sparseml.pytorch.optim import ScheduledModifierManager
    from sparseml.pytorch.utils import ModuleExporter


class SparseMLCallback(Callback):
    """Enables SparseML aware training. Requires a recipe to run during training.

    Args:
        recipe_path: Path to a SparseML compatible yaml recipe.
            More information at https://docs.neuralmagic.com/sparseml/source/recipes.html
    """

    def __init__(self, recipe_path: str):
        if not _SPARSEML_AVAILABLE:
            if not _PL_GREATER_EQUAL_1_4_5:
                raise MisconfigurationException("SparseML requires PyTorch Lightning 1.4.5 or greater.")
            if not _TORCH_MAX_VERSION_SPARSEML:
                raise MisconfigurationException("SparseML requires PyTorch version lower than 1.10.0.")
            raise MisconfigurationException("SparseML has not be installed, install with pip install sparseml")
        self.manager = ScheduledModifierManager.from_yaml(recipe_path)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        optimizer = trainer.optimizers

        if len(optimizer) > 1:
            raise MisconfigurationException("SparseML only supports training with one optimizer.")
        optimizer = optimizer[0]
        optimizer = self.manager.modify(
            pl_module, optimizer, steps_per_epoch=self._num_training_steps_per_epoch(trainer), epoch=0
        )
        trainer.optimizers = [optimizer]

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.manager.finalize(pl_module)

    def _num_training_steps_per_epoch(self, trainer: Trainer) -> int:
        """Total training steps inferred from the datamodule and devices."""
        if isinstance(trainer.limit_train_batches, int) and trainer.limit_train_batches != 0:
            dataset_size = trainer.limit_train_batches
        elif isinstance(trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * trainer.limit_train_batches)
        else:
            dataset_size = len(trainer.datamodule.train_dataloader())

        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)

        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = dataset_size // effective_batch_size

        if trainer.max_steps and trainer.max_steps < max_estimated_steps:
            return trainer.max_steps
        return max_estimated_steps

    @staticmethod
    def export_to_sparse_onnx(
        model: LightningModule, output_dir: str, sample_batch: Optional[torch.Tensor] = None, **export_kwargs: Any
    ) -> None:
        """Exports the model to ONNX format."""
        with model._prevent_trainer_and_dataloaders_deepcopy():
            exporter = ModuleExporter(model, output_dir=output_dir)
            sample_batch = sample_batch if sample_batch is not None else model.example_input_array
            if sample_batch is None:
                raise MisconfigurationException(
                    "To export the model, a sample batch must be passed via "
                    "``SparseMLCallback.export_to_sparse_onnx(model, output_dir, sample_batch=sample_batch)`` "
                    "or an ``example_input_array`` property within the LightningModule"
                )
            exporter.export_onnx(sample_batch=sample_batch, **export_kwargs)
