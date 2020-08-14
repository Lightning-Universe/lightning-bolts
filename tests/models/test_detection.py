import torch
from pl_bolts.models.detection import FasterRCNN


def test_fasterrcnn(tmpdir):
    # NOTE: we probably want to test training, but the detection datasets are quite large
    # so it could be time consuming on the test server

    model = FasterRCNN()

    image = torch.rand(1, 3, 400, 400)
    model(image)
