"""
Here are a VAE and GAN

"""

from pl_bolts.models.autoencoders.basic_ae.basic_ae_module import AE
from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import VAE

from pl_bolts.models.autoencoders.components import resnet18_encoder, resnet18_decoder
from pl_bolts.models.autoencoders.components import resnet50_encoder, resnet50_decoder
