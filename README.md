<div align="center">

<img src="docs/source/_images/logos/bolts_logo.png" width="400px">

**Deep Learning components for extending PyTorch Lightning**

______________________________________________________________________

<p align="center">
  <a href="#install">Installation</a> •
  <a href="https://lightning-bolts.readthedocs.io/en/latest/">Latest Docs</a> •
  <a href="https://lightning-bolts.readthedocs.io/en/stable/">Stable Docs</a> •
  <a href="#what-is-bolts">About</a> •
  <a href="#team">Community</a> •
  <a href="https://www.pytorchlightning.ai/">Website</a> •
  <a href="https://www.grid.ai/">Grid AI</a> •
  <a href="#licence">Licence</a>
</p>

[![PyPI Status](https://badge.fury.io/py/lightning-bolts.svg)](https://badge.fury.io/py/lightning-bolts)
[![PyPI Status](https://pepy.tech/badge/lightning-bolts)](https://pepy.tech/project/lightning-bolts)
[![Build Status](https://dev.azure.com/PytorchLightning/lightning%20Bolts/_apis/build/status/PyTorchLightning.lightning-bolts?branchName=master)](https://dev.azure.com/PytorchLightning/lightning%20Bolts/_build/latest?definitionId=5&branchName=master)
[![codecov](https://codecov.io/gh/PyTorchLightning/lightning-bolts/branch/master/graph/badge.svg)](https://codecov.io/gh/PyTorchLightning/lightning-bolts)
[![CodeFactor](https://www.codefactor.io/repository/github/pytorchlightning/lightning-bolts/badge)](https://www.codefactor.io/repository/github/pytorchlightning/lightning-bolts)

[![Documentation Status](https://readthedocs.org/projects/lightning-bolts/badge/?version=latest)](https://pytorch-lightning-bolts.readthedocs.io/en/latest/)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/lightning-bolts/blob/master/LICENSE)

</div>

______________________________________________________________________

## Continuous Integration

<details>
  <summary>CI testing</summary>

| System / PyTorch ver. |                                                             1.6 (min. req.)                                                              |                                                               1.8 (latest)                                                               |
| :-------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
|    Linux py3.{6,8}    | ![CI full testing](https://github.com/PyTorchLightning/lightning-bolts/workflows/CI%20full%20testing/badge.svg?branch=master&event=push) | ![CI full testing](https://github.com/PyTorchLightning/lightning-bolts/workflows/CI%20full%20testing/badge.svg?branch=master&event=push) |
|     OSX py3.{6,8}     | ![CI full testing](https://github.com/PyTorchLightning/lightning-bolts/workflows/CI%20full%20testing/badge.svg?branch=master&event=push) | ![CI full testing](https://github.com/PyTorchLightning/lightning-bolts/workflows/CI%20full%20testing/badge.svg?branch=master&event=push) |
|    Windows py3.7\*    | ![CI base testing](https://github.com/PyTorchLightning/lightning-bolts/workflows/CI%20base%20testing/badge.svg?branch=master&event=push) | ![CI base testing](https://github.com/PyTorchLightning/lightning-bolts/workflows/CI%20base%20testing/badge.svg?branch=master&event=push) |

- _\* testing just the package itself, we skip full test suite - excluding `tests` folder_

</details>

## Install

Pip / Conda

```bash
pip install lightning-bolts
```

<details>
  <summary>Other installations</summary>

Install bleeding-edge (no guarantees)

```bash
pip install git+https://github.com/PytorchLightning/lightning-bolts.git@master --upgrade
```

In case you want to have full experience you can install all optional packages at once

```bash
pip install lightning-bolts["extra"]
```

</details>

## What is Bolts

Bolts provides a variety of components to extend PyTorch Lightning such as callbacks & datasets, for applied research and production.

## News

- \[2021/08/26\] [Fine-tune Transformers Faster with Lightning Flash and Torch ORT](https://devblog.pytorchlightning.ai/fine-tune-transformers-faster-with-lightning-flash-and-torch-ort-ec2d53789dc3)

#### Example 1: Accelerate Lightning Training with the Torch ORT Callback

TODO: We need something here that is more general than using the transformers package... maybe just an CNN model with the MNISTDataModule?

```python
    from pytorch_lightning import LightningModule, Trainer
    from transformers import AutoModel
    from pl_bolts.callbacks import ORTCallback
    class MyTransformerModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.model = AutoModel.from_pretrained('bert-base-cased')
        ...
    model = MyTransformerModel()
    trainer = Trainer(gpus=1, callbacks=ORTCallback())
    trainer.fit(model)
```

#### Example 2: Lightning SparseML Pruning Callback to accelerate inference

```python
from pl_bolts.models import ImageGPT


class VideoGPT(ImageGPT):
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = _shape_input(x)

        logits = self.gpt(x)
        simclr_features = self.simclr(x)

        # -----------------
        # do something new with GPT logits + simclr_features
        # -----------------

        loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1).long())

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}
```

## Are specific research implementations supported?

We've deprecated a bunch of specific model research, primarily because they've grown outdated or support for them was not possible. This also means in the future, we'll not accept any model specific research. We'd like to encourage users to contribute general component that will help a broad range of problems, however components that help specifics domains will also be welcomed!

For example a tool to help train SSL models as a callback would be accepted, however the next greatest SSL model would be a good contribution to [Lightning Flash](<>).

We've done a better job within [Lightning Flash](<>) to implement SOTA models for applied research. We suggest looking at our [VISSL](<>) Flash integration for SSL based tasks.

See our [deprecated implementations](<>) for more information.

## Contribute!

Bolts is supported by the PyTorch Lightning team and the PyTorch Lightning community!

Join our Slack and/or read our [CONTRIBUTING](./.github/CONTRIBUTING.md) guidelines to get help becoming a contributor!

______________________________________________________________________

## Citation

To cite bolts use:

```
@article{falcon2020framework,
  title={A Framework For Contrastive Self-Supervised Learning And Designing A New Approach},
  author={Falcon, William and Cho, Kyunghyun},
  journal={arXiv preprint arXiv:2009.00104},
  year={2020}
}
```

To cite other contributed models or modules, please cite the authors directly (if they don't have bibtex, ping the authors on a GH issue)

## Licence

Please observe the Apache 2.0 license that is listed in this repository.
In addition the Lightning framework is Patent Pending.
