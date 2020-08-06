# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

### Removed

### Deprecated

## [0.1.0] - yyyy-mm-dd

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



