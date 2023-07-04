import pytest
import torch

from pl_bolts.models.self_supervised.swav.swav_swin import swin_b, swin_s, swin_v2_b, swin_v2_s, swin_v2_t
import torch.nn as nn

model = [swin_s,
        swin_b,
        swin_v2_t,
        swin_v2_s,
        swin_v2_b]


@pytest.mark.parametrize( 'model_architecture, hidden_mlp, prj_head_type , feat_dim',
                         [
                          (swin_s , 0 , nn.Linear , 128),
                          (swin_s , 2048 , nn.Sequential , 128),
                          (swin_b , 0 , nn.Linear , 128),
                          (swin_b , 2048 , nn.Sequential , 128),
                          (swin_v2_t , 0 , nn.Linear , 128),
                          (swin_v2_t , 2048 , nn.Sequential, 128),
                          (swin_v2_s , 0 , nn.Linear , 128),
                          (swin_v2_s , 2048 , nn.Sequential , 128),
                          (swin_v2_b , 0 , nn.Linear , 128),
                          (swin_v2_b , 2048 , nn.Sequential , 128)
                         ]
)
@torch.no_grad()
def test_swin_projection_head(model_architecture, hidden_mlp, prj_head_type , feat_dim):
    model=model_architecture(hidden_mlp = hidden_mlp , output_dim = feat_dim)
    assert isinstance(model.projection_head , prj_head_type)
    
