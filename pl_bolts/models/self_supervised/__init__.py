"""
Self-supervised learning
========================

This module contains models, and losses useful for self-supervised learning research.
These models might also generate better representations than models trained using supervised learning (ie: ResNet-50)
"""
from pl_bolts.models.self_supervised.amdim.module import AMDIM
from pl_bolts.models.self_supervised.cpc.module import CPCV2
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
