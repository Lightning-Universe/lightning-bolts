from pl_bolts.models.self_supervised.cpc.cpc_module import CPCV2
from pl_bolts.models.self_supervised.cpc.networks import cpc_resnet101, cpc_resnet50
from pl_bolts.models.self_supervised.cpc.transforms import (
    CPCTrainTransformsCIFAR10,
    CPCEvalTransformsCIFAR10,
    CPCTrainTransformsSTL10,
    CPCEvalTransformsSTL10,
    CPCTrainTransformsImageNet128,
    CPCEvalTransformsImageNet128
)
