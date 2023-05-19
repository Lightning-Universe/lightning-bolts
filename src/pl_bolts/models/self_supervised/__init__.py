"""These models have been pre-trained using self-supervised learning. The models can also be used without pre-
training and overwritten for your own research.

Here's an example for using these as pretrained models.

.. code-block ::

    from pl_bolts.models.self_supervised import CPC_v2

    images = get_imagenet_batch()

    # extract unsupervised representations
    pretrained = CPC_v2(pretrained=True)
    representations = pretrained(images)

    # use these in classification or any downstream task
    classifications = classifier(representations)
"""
from pl_bolts.models.self_supervised.amdim.amdim_module import AMDIM
from pl_bolts.models.self_supervised.byol.byol_module import BYOL
from pl_bolts.models.self_supervised.cpc.cpc_module import CPC_v2
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from pl_bolts.models.self_supervised.moco.moco2_module import Moco_v2
from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR
from pl_bolts.models.self_supervised.simsiam.simsiam_module import SimSiam
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
from pl_bolts.models.self_supervised.swav.swav_module import SwAV

__all__ = [
    "AMDIM",
    "BYOL",
    "CPC_v2",
    "SSLEvaluator",
    "Moco_v2",
    "SimCLR",
    "SimSiam",
    "SSLFineTuner",
    "SwAV",
]
