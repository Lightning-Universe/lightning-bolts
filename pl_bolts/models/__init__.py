"""
Collection of PyTorchLightning models
"""

from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import VAE
from pl_bolts.models.mnist_module import LitMNIST
from pl_bolts.models.regression import LinearRegression
from pl_bolts.models.regression import LogisticRegression
from pl_bolts.models.vision import PixelCNN
from pl_bolts.models.vision.image_gpt.igpt_module import GPT2, ImageGPT
