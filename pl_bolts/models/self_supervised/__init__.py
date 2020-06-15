"""
Self-supervised learning Models
===============================
Something about this

"""
from pl_bolts.models.self_supervised.amdim.amdim_module import AMDIM
from pl_bolts.models.self_supervised.cpc.cpc_module import CPCV2
from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR
from pl_bolts.models.self_supervised.moco.moco2_module import MocoV2

from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
