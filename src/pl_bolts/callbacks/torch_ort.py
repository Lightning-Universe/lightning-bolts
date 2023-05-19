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
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from pl_bolts.utils import _TORCH_ORT_AVAILABLE

if _TORCH_ORT_AVAILABLE:
    from torch_ort import ORTModule

from pl_bolts.utils.stability import under_review


@under_review()
class ORTCallback(Callback):
    """Enables Torch ORT: Accelerate PyTorch models with ONNX Runtime.

    Wraps a model with the ORT wrapper, lazily converting your module into an ONNX export, to optimize for
    training and inference.

    Usage:

        # via Transformer Tasks
        model = TextClassifier(backbone="facebook/bart-large", num_classes=datamodule.num_classes, enable_ort=True)

        # or via the trainer
        trainer = flash.Trainer(callbacks=ORTCallback())
    """

    def __init__(self) -> None:
        if not _TORCH_ORT_AVAILABLE:
            raise MisconfigurationException(
                "Torch ORT is required to use ORT. See here for installation: https://github.com/pytorch/ort"
            )

    def on_before_accelerator_backend_setup(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not hasattr(pl_module, "model"):
            raise MisconfigurationException(
                "Torch ORT requires to wrap a single model that defines a forward function "
                "assigned as `model` inside the `LightningModule`."
            )
        if not isinstance(pl_module.model, ORTModule):
            pl_module.model = ORTModule(pl_module.model)
