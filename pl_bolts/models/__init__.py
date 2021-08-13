"""Collection of PyTorchLightning models."""
from pl_bolts.models.autoencoders.basic_ae.basic_ae_module import AE
from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import VAE
from pl_bolts.models.mnist_module import LitMNIST
from pl_bolts.models.regression import LinearRegression, LogisticRegression

__all__ = [
    "AE",
    "VAE",
    "LitMNIST",
    "LinearRegression",
    "LogisticRegression",
]
