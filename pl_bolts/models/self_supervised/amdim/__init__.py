from pl_bolts.models.self_supervised.amdim.amdim_module import AMDIM
from pl_bolts.models.self_supervised.amdim.networks import AMDIMEncoder

try:
    from pl_bolts.models.self_supervised.amdim.transforms import (
        AMDIMEvalTransformsCIFAR10,
        AMDIMEvalTransformsImageNet128,
        AMDIMEvalTransformsSTL10,
        AMDIMTrainTransformsCIFAR10,
        AMDIMTrainTransformsImageNet128,
        AMDIMTrainTransformsSTL10,
    )
except ModuleNotFoundError:
    pass
