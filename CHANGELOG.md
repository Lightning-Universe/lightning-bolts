# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.3.3] - 2021-04-17

### Changed

- Suppressed missing package warnings, conditioned by `WARN_MISSING_PACKAGE="1"`  ([#617](https://github.com/PyTorchLightning/lightning-bolts/pull/617))
- Updated all scripts to LARS ([#613](https://github.com/PyTorchLightning/lightning-bolts/pull/613))

### Fixed

- Add missing `dataclass` requirements ([#618](https://github.com/PyTorchLightning/lightning-bolts/pull/618))


## [0.3.2] - 2021-03-20

### Changed

- Renamed SSL modules: `CPCV2` >> `CPC_v2` and `MocoV2` >> `Moco_v2` ([#585](https://github.com/PyTorchLightning/lightning-bolts/pull/585))
- Refactored _setup.py_ to be typing friendly ([#601](https://github.com/PyTorchLightning/lightning-bolts/pull/601))


## [0.3.1] - 2021-03-09

### Added

- Added Pix2Pix model ([#533](https://github.com/PyTorchLightning/lightning-bolts/pull/533))

### Changed

- Moved vision models (`GPT2`, `ImageGPT`, `SemSegment`, `UNet`) to `pl_bolts.models.vision` ([#561](https://github.com/PyTorchLightning/lightning-bolts/pull/561))

### Fixed

- Fixed BYOL moving average update ([#574](https://github.com/PyTorchLightning/lightning-bolts/pull/574))
- Fixed custom gamma in rl ([#550](https://github.com/PyTorchLightning/lightning-bolts/pull/550))
- Fixed PyTorch 1.8 compatibility issue ([#580](https://github.com/PyTorchLightning/lightning-bolts/pull/580),
    [#579](https://github.com/PyTorchLightning/lightning-bolts/pull/579))
- Fixed handling batchnorms in `BatchGradientVerification` ([#569](https://github.com/PyTorchLightning/lightning-bolts/pull/569))
- Corrected `num_rows` calculation in `LatentDimInterpolator` callback ([#573](https://github.com/PyTorchLightning/lightning-bolts/pull/573))


## [0.3.0] - 2021-01-20

### Added

- Added `input_channels` argument to UNet ([#297](https://github.com/PyTorchLightning/lightning-bolts/pull/297))
- Added SwAV ([#239](https://github.com/PyTorchLightning/lightning-bolts/pull/239),
    [#348](https://github.com/PyTorchLightning/lightning-bolts/pull/348),
    [#323](https://github.com/PyTorchLightning/lightning-bolts/pull/323))
- Added data monitor callbacks `ModuleDataMonitor` and `TrainingDataMonitor` ([#285](https://github.com/PyTorchLightning/lightning-bolts/pull/285))
- Added DCGAN module ([#403](https://github.com/PyTorchLightning/lightning-bolts/pull/403)) 
- Added `VisionDataModule` as parent class for `BinaryMNISTDataModule`, `CIFAR10DataModule`, `FashionMNISTDataModule`, 
  and `MNISTDataModule` ([#400](https://github.com/PyTorchLightning/lightning-bolts/pull/400))
- Added GIoU loss ([#347](https://github.com/PyTorchLightning/lightning-bolts/pull/347))
- Added IoU loss ([#469](https://github.com/PyTorchLightning/lightning-bolts/pull/469))
- Added semantic segmentation model `SemSegment` with `UNet` backend ([#259](https://github.com/PyTorchLightning/lightning-bolts/pull/259))
- Added pption to normalize latent interpolation images ([#438](https://github.com/PyTorchLightning/lightning-bolts/pull/438))
- Added flags to datamodules ([#388](https://github.com/PyTorchLightning/lightning-bolts/pull/388))
- Added metric GIoU ([#347](https://github.com/PyTorchLightning/lightning-bolts/pull/347))
- Added Intersection over Union Metric/Loss ([#469](https://github.com/PyTorchLightning/lightning-bolts/pull/469))
- Added SimSiam model ([#407](https://github.com/PyTorchLightning/lightning-bolts/pull/407))
- Added gradient verification callback ([#465](https://github.com/PyTorchLightning/lightning-bolts/pull/465))
- Added Backbones to FRCNN ([#475](https://github.com/PyTorchLightning/lightning-bolts/pull/475))

### Changed

- Decoupled datamodules from models ([#332](https://github.com/PyTorchLightning/lightning-bolts/pull/332),
    [#270](https://github.com/PyTorchLightning/lightning-bolts/pull/270))
- Set PyTorch Lightning 1.0 as the minimum requirement ([#274](https://github.com/PyTorchLightning/lightning-bolts/pull/274))
- Moved `pl_bolts.callbacks.self_supervised.BYOLMAWeightUpdate` to  `pl_bolts.callbacks.byol_updates.BYOLMAWeightUpdate` ([#288](https://github.com/PyTorchLightning/lightning-bolts/pull/288))
- Moved `pl_bolts.callbacks.self_supervised.SSLOnlineEvaluator` to `pl_bolts.callbacks.ssl_online.SSLOnlineEvaluator` ([#288](https://github.com/PyTorchLightning/lightning-bolts/pull/288))
- Moved `pl_bolts.datamodules.*_dataset` to `pl_bolts.datasets.*_dataset` ([#275](https://github.com/PyTorchLightning/lightning-bolts/pull/275))
- Ensured sync across val/test step when using DDP ([#371](https://github.com/PyTorchLightning/lightning-bolts/pull/371))
- Refactored CLI arguments of models ([#394](https://github.com/PyTorchLightning/lightning-bolts/pull/394))
- Upgraded DQN to use `.log` ([#404](https://github.com/PyTorchLightning/lightning-bolts/pull/404))
- Decoupled DataModules from models - CPCV2 ([#386](https://github.com/PyTorchLightning/lightning-bolts/pull/386))
- Refactored datamodules/datasets ([#338](https://github.com/PyTorchLightning/lightning-bolts/pull/338))
- Refactored Vision DataModules ([#400](https://github.com/PyTorchLightning/lightning-bolts/pull/400))
- Refactored `pl_bolts.callbacks` ([#477](https://github.com/PyTorchLightning/lightning-bolts/pull/477))
- Refactored the rest of `pl_bolts.models.self_supervised` ([#481](https://github.com/PyTorchLightning/lightning-bolts/pull/481),
    [#479](https://github.com/PyTorchLightning/lightning-bolts/pull/479)
- Update [`torchvision.utils.make_grid`(https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid)] kwargs to `TensorboardGenerativeModelImageSampler` ([#494](https://github.com/PyTorchLightning/lightning-bolts/pull/494))

### Fixed

- Fixed duplicate warnings when optional packages are unavailable ([#341](https://github.com/PyTorchLightning/lightning-bolts/pull/341))
- Fixed `ModuleNotFoundError` when importing datamoules ([#303](https://github.com/PyTorchLightning/lightning-bolts/pull/303))
- Fixed cyclic imports in `pl_bolts.utils.self_suprvised` ([#350](https://github.com/PyTorchLightning/lightning-bolts/pull/350))
- Fixed VAE loss to use KL term of ELBO ([#330](https://github.com/PyTorchLightning/lightning-bolts/pull/330))
- Fixed dataloders of `MNISTDataModule` to use `self.batch_size` ([#331](https://github.com/PyTorchLightning/lightning-bolts/pull/331))
- Fixed missing `outputs` in SSL hooks for PyTorch Lightning 1.0 ([#277](https://github.com/PyTorchLightning/lightning-bolts/pull/277))
- Fixed stl10 datamodule ([#369](https://github.com/PyTorchLightning/lightning-bolts/pull/369))
- Fixes SimCLR transforms ([#329](https://github.com/PyTorchLightning/lightning-bolts/pull/329))
- Fixed binary MNIST datamodule ([#377](https://github.com/PyTorchLightning/lightning-bolts/pull/377))
- Fixed the end of batch size mismatch ([#389](https://github.com/PyTorchLightning/lightning-bolts/pull/389))
- Fixed `batch_size` parameter for DataModules remaining ([#344](https://github.com/PyTorchLightning/lightning-bolts/pull/344))
- Fixed CIFAR `num_samples` ([#432](https://github.com/PyTorchLightning/lightning-bolts/pull/432))
- Fixed DQN `run_n_episodes` using the wrong environment variable ([#525](https://github.com/PyTorchLightning/lightning-bolts/pull/525))

## [0.2.5] - 2020-10-12

- Enabled PyTorch Lightning 1.0 compatibility

## [0.2.4] - 2020-10-12

- Enabled manual returns ([#267](https://github.com/PyTorchLightning/lightning-bolts/pull/267))

## [0.2.3] - 2020-10-12

### Added

- Enabled PyTorch Lightning 0.10 compatibility ([#264](https://github.com/PyTorchLightning/lightning-bolts/pull/264))
- Added dummy datasets ([#266](https://github.com/PyTorchLightning/lightning-bolts/pull/266))
- Added `KittiDataModule` ([#248](https://github.com/PyTorchLightning/lightning-bolts/pull/248))
- Added `UNet` ([#247](https://github.com/PyTorchLightning/lightning-bolts/pull/247))
- Added reinforcement learning models, losses and datamodules ([#257](https://github.com/PyTorchLightning/lightning-bolts/pull/257))

## [0.2.2] - 2020-09-14

- Fixed confused logit ([#222](https://github.com/PyTorchLightning/lightning-bolts/pull/222))

## [0.2.1] - 2020-09-13

### Added

- Added pretrained VAE with resnet encoders and decoders
- Added pretrained AE with resnet encoders and decoders
- Added CPC pretrained on CIFAR10 and STL10
- Verified BYOL implementation

### Changed

- Dropped all dependencies except PyTorch Lightning and PyTorch
- Decoupled datamodules from GAN ([#206](https://github.com/PyTorchLightning/lightning-bolts/pull/206))
- Modularize AE & VAE ([#196](https://github.com/PyTorchLightning/lightning-bolts/pull/196))

### Fixed

- Fixed gym ([#221](https://github.com/PyTorchLightning/lightning-bolts/pull/221))
- Fix L1/L2 regularization ([#216](https://github.com/PyTorchLightning/lightning-bolts/pull/216))
- Fix max_depth recursion crash in AsynchronousLoader ([#191](https://github.com/PyTorchLightning/lightning-bolts/pull/191))

## [0.2.0] - 2020-09-10

### Added

- Enabled Apache License, Version 2.0

### Changed

- Moved unnecessary dependencies to `__main__` section in BYOL ([#176](https://github.com/PyTorchLightning/lightning-bolts/pull/176))

### Fixed

- Fixed CPC STL10 finetune ([#173](https://github.com/PyTorchLightning/lightning-bolts/pull/173))

## [0.1.1] - 2020-08-23

### Added

- Added Faster RCNN + Pscal VOC DataModule ([#157](https://github.com/PyTorchLightning/lightning-bolts/pull/157))
- Added a better lars scheduling `LARSWrapper` ([#162](https://github.com/PyTorchLightning/lightning-bolts/pull/162))
- Added CPC finetuner ([#158](https://github.com/PyTorchLightning/lightning-bolts/pull/158))
- Added `BinaryMNISTDataModule` ([#153](https://github.com/PyTorchLightning/lightning-bolts/pull/153))
- Added learning rate scheduler to BYOL ([#148](https://github.com/PyTorchLightning/lightning-bolts/pull/148))
- Added Cityscapes DataModule ([#136](https://github.com/PyTorchLightning/lightning-bolts/pull/136))
- Added learning rate scheduler `LinearWarmupCosineAnnealingLR` ([#138](https://github.com/PyTorchLightning/lightning-bolts/pull/138))
- Added BYOL ([#144](https://github.com/PyTorchLightning/lightning-bolts/pull/144))
- Added `ConfusedLogitCallback` ([#118](https://github.com/PyTorchLightning/lightning-bolts/pull/118))
- Added an asynchronous single GPU dataloader. ([#1521](https://github.com/PyTorchLightning/lightning/pull/1521))

### Fixed

- Fixed simclr finetuner ([#165](https://github.com/PyTorchLightning/lightning-bolts/pull/165))
- Fixed STL10 finetuner ([#164](https://github.com/PyTorchLightning/lightning-bolts/pull/164))
- Fixed Image GPT ([#108](https://github.com/PyTorchLightning/lightning-bolts/pull/108))
- Fixed unused MNIST transforms in tran/val/test ([#109](https://github.com/PyTorchLightning/lightning-bolts/pull/109))

### Changed

- Enhanced train batch function ([#107](https://github.com/PyTorchLightning/lightning-bolts/pull/107))

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
