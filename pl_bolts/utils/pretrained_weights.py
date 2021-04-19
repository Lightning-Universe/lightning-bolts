from typing import Optional

from pytorch_lightning import LightningModule

vae_imagenet2012 = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/' \
                   'vae/imagenet_06_22_2019/checkpoints/epoch%3D63.ckpt'

cpcv2_resnet18 = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/' \
                 'cpc/resnet18-v6/epoch%3D85.ckpt'
urls = {'vae-imagenet2012': vae_imagenet2012, 'CPC_v2-resnet18': cpcv2_resnet18}


def load_pretrained(model: LightningModule, class_name: Optional[str] = None) -> None:  # pragma: no cover
    if class_name is None:
        class_name = model.__class__.__name__
    ckpt_url = urls[class_name]
    weights_model = model.__class__.load_from_checkpoint(ckpt_url)
    model.load_state_dict(weights_model.state_dict())
