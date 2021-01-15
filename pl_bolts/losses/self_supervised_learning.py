import numpy as np
import torch
from torch import nn

from pl_bolts.models.vision.pixel_cnn import PixelCNN


def nt_xent_loss(out_1, out_2, temperature):
    """
    Loss used in SimCLR
    """
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    # Negative similarity
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity :
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / neg).mean()

    return loss


class CPCTask(nn.Module):
    """
    Loss used in CPC
    """

    def __init__(self, num_input_channels, target_dim=64, embed_scale=0.1):
        super().__init__()
        self.target_dim = target_dim
        self.embed_scale = embed_scale

        self.target_cnn = torch.nn.Conv2d(num_input_channels, self.target_dim, kernel_size=1)
        self.pred_cnn = torch.nn.Conv2d(num_input_channels, self.target_dim, kernel_size=1)
        self.context_cnn = PixelCNN(num_input_channels)

    def compute_loss_h(self, targets, preds, i):
        b, c, h, w = targets.shape

        # (b, c, h, w) -> (num_vectors, emb_dim)
        # every vector (c-dim) is a target
        targets = targets.permute(0, 2, 3, 1).contiguous().reshape([-1, c])

        # select the future (south) targets to predict
        # selects all of the ones south of the current source
        preds_i = preds[:, :, :-(i + 1), :] * self.embed_scale

        # (b, c, h, w) -> (b*w*h, c) (all features)
        # this ordering matches the targets
        preds_i = preds_i.permute(0, 2, 3, 1).contiguous().reshape([-1, self.target_dim])

        # calculate the strength scores
        logits = torch.matmul(preds_i, targets.transpose(-1, -2))

        # generate the labels
        n = b * (h - i - 1) * w
        b1 = torch.arange(n) // ((h - i - 1) * w)
        c1 = torch.arange(n) % ((h - i - 1) * w)
        labels = b1 * h * w + (i + 1) * w + c1
        labels = labels.to(logits.device)
        labels = labels.long()

        loss = nn.functional.cross_entropy(logits, labels)
        return loss

    def forward(self, Z):
        losses = []

        context = self.context_cnn(Z)
        targets = self.target_cnn(Z)

        _, _, h, w = Z.shape

        # future prediction
        preds = self.pred_cnn(context)
        for steps_to_ignore in range(h - 1):
            for i in range(steps_to_ignore + 1, h):
                loss = self.compute_loss_h(targets, preds, i)
                if not torch.isnan(loss):
                    losses.append(loss)

        loss = torch.stack(losses).sum()
        return loss


class AmdimNCELoss(nn.Module):
    """
    Compute the NCE scores for predicting r_src->r_trg.
    """

    def __init__(self, tclip):
        super().__init__()
        self.tclip = tclip

    def forward(self, anchor_representations, positive_representations, mask_mat):
        """
        Args:
            anchor_representations: (batch_size, emb_dim)
            positive_representations: (emb_dim, n_batch * w* h) (ie: nb_feat_vectors x embedding_dim)
            mask_mat: (n_batch_gpu, n_batch)

        Output:
            raw_scores: (n_batch_gpu, n_locs)
            nce_scores: (n_batch_gpu, n_locs)
            lgt_reg : scalar
        """
        r_src = anchor_representations
        r_trg = positive_representations

        # RKHS = embedding dim
        batch_size, emb_dim = r_src.size()
        nb_feat_vectors = r_trg.size(1) // batch_size

        # (b, b) -> (b, b, nb_feat_vectors)
        # all zeros with ones in diagonal tensor... (ie: b1 b1 are all 1s, b1 b2 are all zeros)
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, nb_feat_vectors).float()

        # negative mask
        mask_neg = 1. - mask_pos

        # -------------------------------
        # ALL SCORES COMPUTATION
        # compute src->trg raw scores for batch
        # (b, dim) x (dim, nb_feats*b) -> (b, b, nb_feats)
        # vector for each img in batch times all the vectors of all images in batch
        raw_scores = torch.mm(r_src, r_trg).float()
        raw_scores = raw_scores.reshape(batch_size, batch_size, nb_feat_vectors)

        # -----------------------
        # STABILITY TRICKS
        # trick 1: weighted regularization term
        raw_scores = raw_scores / emb_dim**0.5
        lgt_reg = 5e-2 * (raw_scores**2.).mean()

        # trick 2: tanh clip
        raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)
        """
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        """
        # ----------------------
        # EXTRACT POSITIVE SCORES
        # use the index mask to pull all the diagonals which are b1 x b1
        # (batch_size, nb_feat_vectors)
        pos_scores = (mask_pos * raw_scores).sum(dim=1)

        # ----------------------
        # EXTRACT NEGATIVE SCORES
        # pull everything except diagonal and apply clipping
        # (batch_size, batch_size, nb_feat_vectors)
        # diagonals have - clip vals. everything else has actual negative stores
        neg_scores = (mask_neg * raw_scores) - (self.tclip * mask_pos)

        # (batch_size, batch_size * nb_feat_vectors) -> (batch_size, batch_size, nb_feat_vectors)
        neg_scores = neg_scores.reshape(batch_size, -1)
        mask_neg = mask_neg.reshape(batch_size, -1)

        # ---------------------
        # STABLE SOFTMAX
        # max for each row of negative samples
        # will use max in safe softmax
        # (n_batch_gpu, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]

        # DENOMINATOR
        # sum over only negative samples (none from the diagonal)
        neg_sumexp = (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)

        # NUMERATOR
        # compute numerators for the NCE log-softmaxes
        pos_shiftexp = pos_scores - neg_maxes

        # FULL NCE
        nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = -nce_scores.mean()

        return nce_scores, lgt_reg


class FeatureMapContrastiveTask(nn.Module):
    """
    Performs an anchor, positive negative pair comparison for each each tuple of feature maps passed.

    .. code-block:: python

        # extract feature maps
        pos_0, pos_1, pos_2 = encoder(x_pos)
        anc_0, anc_1, anc_2 = encoder(x_anchor)

        # compare only the 0th feature maps
        task = FeatureMapContrastiveTask('00')
        loss, regularizer = task((pos_0), (anc_0))

        # compare (pos_0 to anc_1) and (pos_0, anc_2)
        task = FeatureMapContrastiveTask('01, 02')
        losses, regularizer = task((pos_0, pos_1, pos_2), (anc_0, anc_1, anc_2))
        loss = losses.sum()

        # compare (pos_1 vs a anc_random)
        task = FeatureMapContrastiveTask('0r')
        loss, regularizer = task((pos_0, pos_1, pos_2), (anc_0, anc_1, anc_2))

    .. code-block:: python

        # with bidirectional the comparisons are done both ways
        task = FeatureMapContrastiveTask('01, 02')

        # will compare the following:
        # 01: (pos_0, anc_1), (anc_0, pos_1)
        # 02: (pos_0, anc_2), (anc_0, pos_2)
    """

    def __init__(self, comparisons: str = '00, 11', tclip: float = 10.0, bidirectional: bool = True):
        """
        Args:
            comparisons: groupings of feature map indices to compare (zero indexed, 'r' means random) ex: '00, 1r'
            tclip: stability clipping value
            bidirectional: if true, does the comparison both ways
        """
        super().__init__()
        self.tclip = tclip
        self.bidirectional = bidirectional
        self.map_indexes = self.parse_map_indexes(comparisons)
        self.comparisons = comparisons
        self.masks = {}
        self.nce_loss = AmdimNCELoss(tclip)

    @staticmethod
    def parse_map_indexes(comparisons):
        """
        Example::

            >>> FeatureMapContrastiveTask.parse_map_indexes('11')
            [(1, 1)]
            >>> FeatureMapContrastiveTask.parse_map_indexes('11,59')
            [(1, 1), (5, 9)]
            >>> FeatureMapContrastiveTask.parse_map_indexes('11,59, 2r')
            [(1, 1), (5, 9), (2, -1)]
        """
        map_indexes = [x.strip() for x in comparisons.split(',')]
        for tup_i in range(len(map_indexes)):
            (a, b) = map_indexes[tup_i]
            if a == 'r':
                a = '-1'
            if b == 'r':
                b = '-1'
            map_indexes[tup_i] = (int(a), int(b))

        return map_indexes

    def feat_size_w_mask(self, w, feature_map):
        masks_r5 = torch.zeros((w, w, 1, w, w), device=feature_map.device).type(torch.bool)
        for i in range(w):
            for j in range(w):
                masks_r5[i, j, 0, i, j] = 1
        masks_r5 = masks_r5.reshape(-1, 1, w, w)
        return masks_r5

    def _sample_src_ftr(self, r_cnv, masks):
        # get feature dimensions
        n_batch = r_cnv.size(0)
        feat_dim = r_cnv.size(1)

        if masks is not None:
            # subsample from conv-ish r_cnv to get a single vector
            mask_idx = torch.randint(0, masks.size(0), (n_batch, ), device=r_cnv.device)
            mask = masks[mask_idx]
            r_cnv = torch.masked_select(r_cnv, mask)

        # flatten features for use as globals in glb->lcl nce cost
        r_vec = r_cnv.reshape(n_batch, feat_dim)
        return r_vec

    def __cache_dimension_masks(self, *args):
        # cache masks for each feature map we'll need
        if len(self.masks) == 0:
            for m1 in args:
                batch_size, emb_dim, h, w = m1.size()

                # make mask
                if h not in self.masks:
                    mask = self.feat_size_w_mask(h, m1)
                    self.masks[h] = mask

    def __compare_maps(self, m1, m2):
        b, c, h, w = m1.size()

        mask_1 = self.masks[h]
        src = self._sample_src_ftr(m1, mask_1)

        # target vectors
        # (b, c, h, w) -> (c, b * h * w)
        tgt = m2.permute(1, 0, 2, 3).reshape(c, -1)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(b, device=m1.device)

        # compare
        loss, regularizer = self.nce_loss(src, tgt, diag_mat)

        return loss, regularizer

    def forward(self, anchor_maps, positive_maps):
        """
        Takes in a set of tuples, each tuple has two feature maps with all matching dimensions

        Example:

            >>> import torch
            >>> from pytorch_lightning import seed_everything
            >>> seed_everything(0)
            0
            >>> a1 = torch.rand(3, 5, 2, 2)
            >>> a2 = torch.rand(3, 5, 2, 2)
            >>> b1 = torch.rand(3, 5, 2, 2)
            >>> b2 = torch.rand(3, 5, 2, 2)
            ...
            >>> task = FeatureMapContrastiveTask('01, 11')
            ...
            >>> losses, regularizer = task((a1, a2), (b1, b2))
            >>> losses
            tensor([2.2351, 2.1902])
            >>> regularizer
            tensor(0.0324)
        """
        assert len(anchor_maps) == len(self.map_indexes), f'expected each input to have {len(self.map_indexes)} tensors'

        self.__cache_dimension_masks(*(anchor_maps + positive_maps))

        regularizer = 0
        losses = []
        for (ai, pi) in self.map_indexes:

            # choose a random map
            if ai == -1:
                ai = np.random.randint(0, len(anchor_maps))
            if pi == -1:
                pi = np.random.randint(0, len(anchor_maps))

            # pull out the maps
            anchor = anchor_maps[ai]
            pos = positive_maps[pi]

            # m1 vs m2
            loss1, reg1 = self.__compare_maps(anchor, pos)
            map_reg = reg1
            map_loss = loss1

            # add second direction if requested
            if self.bidirectional:
                # swap maps
                anchor = positive_maps[ai]
                pos = anchor_maps[pi]

                loss2, reg2 = self.__compare_maps(anchor, pos)
                map_reg = 0.5 * (reg1 + reg2)
                map_loss = 0.5 * (loss1 + loss2)

            regularizer += map_reg
            losses.append(map_loss.mean())

        return torch.stack(losses), regularizer


def tanh_clip(x, clip_val=10.):
    """
    soft clip values to the range [-clip_val, +clip_val]
    """
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip
