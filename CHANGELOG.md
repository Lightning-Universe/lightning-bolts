# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - YYYY-MM-DD

### Added

- Added `input_channels` argument to UNet ([#297](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/297))

- Added SwAV ([#239](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/239), [#348](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/348), [#323](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/323))

- Added data monitor callbacks `ModuleDataMonitor` and `TrainingDataMonitor` ([#285](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/285))

### Changed

- Decoupled datamodules from models ([#332](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/332), [#270](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/270))

- Set PyTorch Lightning 1.0 as the minimum requirement ([#274](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/274))

- Move `pl_bolts.callbacks.self_supervised.BYOLMAWeightUpdate` to  `pl_bolts.callbacks.byol_updates.BYOLMAWeightUpdate` ([#288](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/288))

- Move `pl_bolts.callbacks.self_supervised.SSLOnlineEvaluator` to `pl_bolts.callbacks.ssl_online.SSLOnlineEvaluator` ([#288](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/288))

- Move `pl_bolts.datamodules.*_dataset` to `pl_bolts.datasets.*_dataset` ([#275](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/275))

### Fixed

- Fixed duplicate warnings when optional packages are unavailable ([#341](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/341))

- Fixed ModuleNotFoundError when importing datamoules ([#303](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/303))

- Fixed cyclic imports in `pl_bolts.utils.self_suprvised` ([#350](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/350))

- Fixed VAE loss to use KL term of ELBO ([#330](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/330))

- Fixed dataloders of `MNISTDataModule` to use `self.batch_size` ([#331](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/331))

- Fixed missing `outputs` in SSL hooks for PyTorch Lightning 1.0 ([#277](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/277))

### Removed

### Deprecated

## [0.2.5] - 2020-10-12

### Added

- Enabled PyTorch Lightning 1.0 compatibility

## [0.2.4] - 2020-10-12

## [0.2.3] - 2020-10-12

### Added

- Enabled PyTorch Lightning 0.10 compatibility ([#264]())
- Added dummy datasets ([#266](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/266))
- Added semantic segmentation model `SemSegment` with `UNet` backend ([#259](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/259))
- Added `KittiDataModule` ([#248](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/248))
- Added `UNet` ([#247](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/247))
- Added reinforcement learning models, losses and datamodules ([#257](https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/257))

## [0.2.2] - 2020-09-14

## [0.2.1] - 2020-09-13

### Added

- Added pretrained VAE with resnet encoders and decoders
- Added pretrained AE with resnet encoders and decoders
- Added CPC pretrained on CIFAR10 and STL10
- Verified BYOL implementation

### Changed

- Dropped all dependencies except PyTorch Lightning and PyTorch

## [0.2.0] - 2020-09-10

## [0.1.1] - 2020-08-23

## [0.1.0] - 2020-07-02

### Added

- Added setup and repo structure
- Added requirements
- Added docs
- Added Manifest
- Added coverage
- Added MNIST template
- Added VAE template
- Added GAN + AE + MNIST
- Added Linear Regression
- Added Moco2g
- Added simclr
- Added RL module
- Added Loggers
- Added Transforms
- Added Tiny Datasets
- Added regularization to linear + logistic models
- Added Linear and Logistic Regression tests
- Added Image GPT
- Added Recommenders module
- Added an asynchronous single GPU dataloader. ([#1521](https://github.com/PyTorchLightning/pytorch-lightning/pull/1521))

### Changed

- Device is no longer set in the DQN model init
- Moved RL loss function to the losses module
- Moved rl.common.experience to datamodules
- train_batch function to VPG model to generate batch of data at each step (POC)
- Experience source no longer gets initialized with a device, instead the device is passed at each step()
- Refactored ExperienceSource classes to be handle multiple environments. 

### Removed

- Removed N-Step DQN as the latest version of the DQN supports N-Step by setting the `n_step` arg to n
- Deprecated common.experience

### Fixed

- Documentation 
- Doct tests
- CI pipeline
- Imports and pkg
- CPC fixes



