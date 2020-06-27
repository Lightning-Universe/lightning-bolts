import torch
from pl_bolts.models import GPT2


def test_gpt2(tmpdir):

    # TODO: should there be a "correctness" test as well
    seq_len = 17
    batch_size = 32
    classes = 10
    x = torch.randint(0, 10, (seq_len, batch_size))

    model = GPT2(embed_dim=16, heads=2, layers=2, num_positions=28*28, vocab_size=16, num_classes=classes)
    model(x)