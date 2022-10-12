import torch

from pl_bolts.models.detection.yolo.target_matching import _sim_ota_match


def test_sim_ota_match(catch_warnings):
    # IoUs will determined that 2 and 1 predictions will be selected for the first and the second target.
    ious = torch.tensor([[0.1, 0.1, 0.9, 0.9], [0.2, 0.3, 0.4, 0.1]])
    # Costs will determine that the first and the last prediction will be selected for the first target, and the first
    # prediction will be selected for the second target. Since the first prediction was selected for both targets, it
    # will be matched to the best target only (the second one).
    costs = torch.tensor([[0.3, 0.5, 0.4, 0.3], [0.1, 0.2, 0.5, 0.3]])
    matched_preds, matched_targets = _sim_ota_match(costs, ious)
    assert len(matched_preds) == 4
    assert matched_preds[0]
    assert not matched_preds[1]
    assert not matched_preds[2]
    assert matched_preds[3]
    assert len(matched_targets) == 2  # Two predictions were matched.
    assert matched_targets[0] == 1  # Which target was matched to the first prediction.
    assert matched_targets[1] == 0  # Which target was matched to the last prediction.
