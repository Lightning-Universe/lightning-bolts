import pytorch_lightning as pl
import torch
from torch import nn as nn


class Block(nn.Module):

    def __init__(self, embed_dim, heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        attn_mask = torch.full((len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT2(pl.LightningModule):
    """
    GPT-2 from `language Models are Unsupervised Multitask Learners <https://d4mucfpksywv.cloudfront.net/
    better-language-models/language-models.pdf>`_

    Paper by:  Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever

    Implementation contributed by:

        - `Teddy Koker <https://github.com/teddykoker>`_

    Example::

        from pl_bolts.models import GPT2

        seq_len = 17
        batch_size = 32
        vocab_size = 16
        x = torch.randint(0, vocab_size, (seq_len, batch_size))
        model = GPT2(embed_dim=32, heads=2, layers=2, num_positions=seq_len, vocab_size=vocab_size, num_classes=4)
        results = model(x)
    """

    def __init__(
        self,
        embed_dim: int,
        heads: int,
        layers: int,
        num_positions: int,
        vocab_size: int,
        num_classes: int,
    ):
        super(GPT2, self).__init__()
        self.save_hyperparameters()

        self._init_sos_token()
        self._init_embeddings()
        self._init_layers()

    def _init_sos_token(self):
        self.sos = torch.nn.Parameter(torch.zeros(self.hparams.embed_dim))
        nn.init.normal_(self.sos)

    def _init_embeddings(self):
        self.token_embeddings = nn.Embedding(self.hparams.vocab_size, self.hparams.embed_dim)
        self.position_embeddings = nn.Embedding(self.hparams.num_positions, self.hparams.embed_dim)

    def _init_layers(self):
        self.layers = nn.ModuleList()
        for _ in range(self.hparams.layers):
            self.layers.append(Block(self.hparams.embed_dim, self.hparams.heads))

        self.ln_f = nn.LayerNorm(self.hparams.embed_dim)
        self.head = nn.Linear(self.hparams.embed_dim, self.hparams.vocab_size, bias=False)
        self.clf_head = nn.Linear(self.hparams.embed_dim, self.hparams.num_classes)

    def forward(self, x, classify=False):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        length, batch = x.shape

        h = self.token_embeddings(x.long())

        # prepend sos token
        sos = torch.ones(1, batch, self.hparams.embed_dim, device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], axis=0)

        # add positional embeddings
        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)

        # transformer
        for layer in self.layers:
            h = layer(h)

        if not classify:
            # return logits
            return self.head(h)

        h = torch.mean(h, dim=0)  # average pool over sequence
        return self.clf_head(h)  # return classification logits
