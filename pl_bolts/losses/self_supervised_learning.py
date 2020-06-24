import numpy as np
import torch
from torch import nn

from pl_bolts.models.vision import PixelCNN


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
    def __init__(self, tclip):
        super().__init__()
        self.tclip = tclip

    def forward(self, anchor_representations, positive_representations, mask_mat):
        """
        Compute the NCE scores for predicting r_src->r_trg.
        Args:
          anchor_representations   : (batch_size, emb_dim)
          positive_representations : (emb_dim, n_batch * w* h) (ie: nb_feat_vectors x embedding_dim)
          mask_mat                 : (n_batch_gpu, n_batch)

        Output:
          raw_scores : (n_batch_gpu, n_locs)
          nce_scores : (n_batch_gpu, n_locs)
          lgt_reg    : scalar
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
        raw_scores = raw_scores / emb_dim ** 0.5
        lgt_reg = 5e-2 * (raw_scores ** 2.).mean()

        # trick 2: tanh clip
        raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)

        '''
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        '''
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

    def __init__(self, tclip=10., bidirectional=True):
        """
        Performs an anchor, positive negative pair comparison for each each tuple of feature maps passed in

        Example::

            # extract feature maps
            pos_0, pos_1, pos_2 = encoder(x_pos)
            anc_0, anc_1, anc_2 = encoder(x_anchor)

            task = FeatureMapContrastiveTask()
            g0 = (pos_0, anc_0)
            g1 = (pos_1, anc_1)
            g2 = (pos_2, anc_2)
            (loss_0, loss_1, loss_2), regularizer = task(g0, g1, g2)

            # each loss_x corresponds to
            # if not bidirectional only loss_anc_pos is calculated
            loss_anc_pos = phi(pos_0, anc_0)
            loss_pos_anc = phi(anc_0, pos_0)
            g0 = loss_anc_pos + loss_pos_anc

        Arg:

            pattern: groupings of feature map indices to compare
            tclip: stability clipping value
            bidirectional: if true, does the comparison both ways

        """
        super().__init__()
        self.tclip = tclip
        self.bidirectional = bidirectional
        self.masks = {}
        self.nce_loss = AmdimNCELoss(tclip)

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
            mask_idx = torch.randint(0, masks.size(0), (n_batch,), device=r_cnv.device)
            mask = masks[mask_idx]
            r_cnv = torch.masked_select(r_cnv, mask)

        # flatten features for use as globals in glb->lcl nce cost
        r_vec = r_cnv.reshape(n_batch, feat_dim)
        return r_vec

    def __cache_dimension_masks(self, *args):
        # cache masks for each feature map we'll need
        if len(self.masks) == 0:
            for (m1, me) in args:
                batch_size, emb_dim, h, w = m1.size()

                # make mask
                if h not in self.masks:
                    mask = self.feat_size_w_mask(h, m1)
                    self.masks[h] = mask

    def forward(self, *args):
        """
        Takes in a set of tuples, each tuple has two feature maps with all matching dimensions
        """
        self.__cache_dimension_masks(*args)

        regularizer = 0
        losses = []
        for (m1, m2) in args:
            b, c, h, w = m1.size()
            b2, c2, h2, w2 = m2.size()

            assert b == b2
            assert c == c2
            assert h == h2
            assert w == w2

            # get source vectors
            mask_1 = self.masks[h]
            src_1 = self._sample_src_ftr(m1, mask_1)
            src_2 = self._sample_src_ftr(m2, mask_1)

            # target vectors
            # (b, c, h, w) -> (c, b * h * w)
            tgt_1 = m1.permute(1, 0, 2, 3).reshape(c, -1)
            tgt_2 = m2.permute(1, 0, 2, 3).reshape(c, -1)

            # make masking matrix to help compute nce costs
            # (b x b) zero matrix with 1s in the diag
            diag_mat = torch.eye(b, device=m1.device)

            # (m1 vs m2)
            loss_12, regularizer_12 = self.nce_loss(src_1, tgt_2, diag_mat)

            # (m2 vs m1)
            loss_21, regularizer_21 = self.nce_loss(src_2, tgt_1, diag_mat)

            # map_loss
            map_loss = 0.5 * (loss_12, loss_21)
            losses.append(map_loss.mean())

            # regularizer
            regularizer += regularizer_12
            regularizer += regularizer_21

        # scale regularizer
        regularizer = regularizer * 0.5

        return losses, regularizer


class AMDIM_11_55_77_ContrastiveTask(FeatureMapContrastiveTask):


    def __init__(self, tclip: float = 10.0, bidirectional: bool = True):
        """
        Compares three sets of feature maps

        Example::

            # pseudocode!
            p1, p5, p7 = encoder(x_pos)
            a1, a5, a7 = encoder(x_anchor)

            phi = lambda a, b: mm(a, b)
            pa_loss = phi(p1, a1) + phi(p5, a5) + phi(p7, a7)

            # if bidirectional also calculate the other way
            ap_loss = phi(a1, p1) + phi(a5, p5) + phi(a7, p7)

            loss = pa_loss + ap_loss

        Arg:

            tclip: stability clipping value
            bidirectional: if true, does the comparison both ways
        """
        super().__init__(tclip, bidirectional)

    def contrastive_task(self, x1_maps, x2_maps):
        r1_x1, r5_x1, r7_x1 = x1_maps
        r1_x2, r5_x2, r7_x2 = x2_maps

        batch_size, emb_dim, _, _ = r1_x1.size()

        # -----------------
        # SOURCE VECTORS
        # 1 feature vector per image per feature map location
        # img 1
        b_1, e_1, h_1, w_1 = r1_x1.size()
        mask_1 = self.masks[h_1]
        r1_src_1 = self._sample_src_ftr(r1_x1, mask_1)
        r1_src_2 = self._sample_src_ftr(r1_x2, mask_1)

        # img 2
        b_5, e_5, h_5, w_5 = r5_x1.size()
        mask_5 = self.masks[h_5]
        r5_src_1 = self._sample_src_ftr(r5_x1, mask_5)
        r5_src_2 = self._sample_src_ftr(r5_x2, mask_5)

        b_7, e_7, h_7, w_7 = r7_x1.size()
        mask_7 = self.masks[h_7]
        r7_src_1 = self._sample_src_ftr(r7_x1, mask_7)
        r7_src_2 = self._sample_src_ftr(r7_x2, mask_7)

        # -----------------
        # TARGET VECTORS
        # before shape: (n_batch, emb_dim, w, h)
        r1_trg_1 = r1_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r5_trg_1 = r5_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r7_trg_1 = r7_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        r1_trg_2 = r1_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r5_trg_2 = r5_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r7_trg_2 = r7_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        # after shape: (emb_dim, n_batch * w * h)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_x1.device.index)

        # -----------------
        # NCE COSTS
        # compute costs for 1->1 prediction
        # use last layer to predict the layer with (5x5 features)
        loss_1t1_1, regularizer_1t1_1 = self.nce_loss(r1_src_1, r1_trg_2, diag_mat)  # img 1
        loss_1t1_2, regularizer_1t1_2 = self.nce_loss(r1_src_2, r1_trg_1, diag_mat)  # img 2

        # compute costs for 5->5 prediction
        # use last layer to predict the layer with (7x7 features)
        loss_5t5_1, regularizer_1t7_1 = self.nce_loss(r5_src_1, r5_trg_2, diag_mat)  # img 1
        loss_5t5_2, regularizer_1t7_2 = self.nce_loss(r5_src_2, r5_trg_1, diag_mat)  # img 2

        # compute costs for 7->7 prediction
        # use (5x5) layer to predict the (5x5) layer
        loss_7t7_1, regularizer_7t7_1 = self.nce_loss(r7_src_1, r7_trg_2, diag_mat)  # img 1
        loss_7t7_2, regularizer_7t7_2 = self.nce_loss(r7_src_2, r7_trg_1, diag_mat)  # img 2

        # combine costs for optimization
        loss_1t1 = 0.5 * (loss_1t1_1 + loss_1t1_2)
        loss_5t5 = 0.5 * (loss_5t5_1 + loss_5t5_2)
        loss_7t7 = 0.5 * (loss_7t7_1 + loss_7t7_2)

        # regularizer
        regularizer = 0.5 * (regularizer_1t1_1 + regularizer_1t1_2 +
                             regularizer_1t7_1 + regularizer_1t7_2 +
                             regularizer_7t7_1 + regularizer_7t7_2)

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss_1t1 = loss_1t1.mean()
        loss_5t5 = loss_5t5.mean()
        loss_7t7 = loss_7t7.mean()
        regularizer = regularizer.mean()
        return loss_1t1 + loss_5t5 + loss_7t7, regularizer


class ContrastiveTask_15_17_55(FeatureMapContrastiveTask):

    def __init__(self, tclip=10.0):
        """
        This task is used in AMDIM
        (`Philip Bachman, R Devon Hjelm, William Buchwalter <https://arxiv.org/abs/1906.00910>`_).

        This implementation is adapted from the `original repo <https://github.com/Philip-Bachman/amdim-public>`_.

        .. code-block:: python

            # pseudocode
            phi = lambda a, b: mat_mul(a, b)

            loss15 = phi(f1, g5)
            loss17 = phi(f1, g7)
            loss55 = phi(f5, g5)

            total_loss = loss15 + loss17 + loss55

        To use this task, pass in two sets of feature maps

        Example::

            task = ContrastiveTask_15_17_55()

            # 3 feature maps per image
            # each feature map is (batch, channels, n, n) where fn, gn
            # (ie: f5 = (b, c, 5, 5))
            f1, f5, f7 = encoder(x_pos)
            g1, g5, g7 = encoder(x_anchor)

            loss, regularizer = task(x1_maps=(f1, f5, f7), x2_maps=(g1, g5, g7))

        """
        super().__init__(tclip)

    def contrastive_task(self, x1_maps, x2_maps):
        r1_x1, r5_x1, r7_x1 = x1_maps
        r1_x2, r5_x2, r7_x2 = x2_maps

        batch_size, emb_dim, _, _ = r1_x1.size()

        # -----------------
        # SOURCE VECTORS
        # 1 feature vector per image per feature map location
        # img 1
        b_1, e_1, h_1, w_1 = r1_x1.size()
        mask_1 = self.masks[h_1]
        r1_src_1 = self._sample_src_ftr(r1_x1, mask_1)
        r1_src_2 = self._sample_src_ftr(r1_x2, mask_1)

        # img 2
        b_5, e_5, h_5, w_5 = r5_x1.size()
        mask_5 = self.masks[h_5]
        r5_src_1 = self._sample_src_ftr(r5_x1, mask_5)
        r5_src_2 = self._sample_src_ftr(r5_x2, mask_5)

        b_7, e_7, h_7, w_7 = r7_x1.size()
        mask_7 = self.masks[h_7]
        r7_src_1 = self._sample_src_ftr(r7_x1, mask_7)
        r7_src_2 = self._sample_src_ftr(r7_x2, mask_7)

        # -----------------
        # TARGET VECTORS
        # before shape: (n_batch, emb_dim, w, h)
        r1_trg_1 = r1_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r5_trg_1 = r5_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r7_trg_1 = r7_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        r1_trg_2 = r1_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r5_trg_2 = r5_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r7_trg_2 = r7_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        # after shape: (emb_dim, n_batch * w * h)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_x1.device.index)

        # -----------------
        # NCE COSTS
        # compute costs for 1->1 prediction
        # use last layer to predict the layer with (5x5 features)
        loss_1t1_1, regularizer_1t1_1 = self.nce_loss(r1_src_1, r1_trg_2, diag_mat)  # img 1
        loss_1t1_2, regularizer_1t1_2 = self.nce_loss(r1_src_2, r1_trg_1, diag_mat)  # img 2

        # compute costs for 5->5 prediction
        # use last layer to predict the layer with (7x7 features)
        loss_5t5_1, regularizer_1t7_1 = self.nce_loss(r5_src_1, r5_trg_2, diag_mat)  # img 1
        loss_5t5_2, regularizer_1t7_2 = self.nce_loss(r5_src_2, r5_trg_1, diag_mat)  # img 2

        # compute costs for 7->7 prediction
        # use (5x5) layer to predict the (5x5) layer
        loss_7t7_1, regularizer_7t7_1 = self.nce_loss(r7_src_1, r7_trg_2, diag_mat)  # img 1
        loss_7t7_2, regularizer_7t7_2 = self.nce_loss(r7_src_2, r7_trg_1, diag_mat)  # img 2

        # combine costs for optimization
        loss_1t1 = 0.5 * (loss_1t1_1 + loss_1t1_2)
        loss_5t5 = 0.5 * (loss_5t5_1 + loss_5t5_2)
        loss_7t7 = 0.5 * (loss_7t7_1 + loss_7t7_2)

        # regularizer
        regularizer = 0.5 * (regularizer_1t1_1 + regularizer_1t1_2 +
                             regularizer_1t7_1 + regularizer_1t7_2 +
                             regularizer_7t7_1 + regularizer_7t7_2)

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss_1t1 = loss_1t1.mean()
        loss_5t5 = loss_5t5.mean()
        loss_7t7 = loss_7t7.mean()
        regularizer = regularizer.mean()
        return loss_1t1 + loss_5t5 + loss_7t7, regularizer


class AMDIM_1Random_ContrastiveTask(FeatureMapContrastiveTask):

    def contrastive_task(self, x1_maps, x2_maps):
        r1_x1, r5_x1, r7_x1 = x1_maps
        r1_x2, r5_x2, r7_x2 = x2_maps

        batch_size, emb_dim, _, _ = r1_x1.size()

        # -----------------
        # SOURCE VECTORS
        # 1 feature vector per image per feature map location
        # img 1
        b_1, e_1, h_1, w_1 = r1_x1.size()
        mask_1 = self.masks[h_1]

        r1_src_1 = self._sample_src_ftr(r1_x1, mask_1)
        r1_src_2 = self._sample_src_ftr(r1_x2, mask_1)

        # pick the target map
        target_map_idx = np.random.randint(0, len(x2_maps), 1)[0]
        target_map_x2 = x2_maps[target_map_idx]
        target_map_x1 = x1_maps[target_map_idx]

        # -----------------
        # TARGET VECTORS
        # before shape: (n_batch, emb_dim, w, h)
        target_map_x2 = target_map_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        target_map_x1 = target_map_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_x1.device.index)

        # -----------------
        # NCE COSTS
        # compute costs for 1->5 prediction
        # use last layer to predict the layer with (5x5 features)
        loss_1tR_1, regularizer_1t5_1 = self.nce_loss(r1_src_1, target_map_x2, diag_mat)  # img 1
        loss_1tR_2, regularizer_1t5_2 = self.nce_loss(r1_src_2, target_map_x1, diag_mat)  # img 2

        # combine costs for optimization
        loss_1tR = 0.5 * (loss_1tR_1 + loss_1tR_2)

        # regularizer
        regularizer = 0.5 * (regularizer_1t5_1 + regularizer_1t5_2)

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss_1tR = loss_1tR.mean()
        regularizer = regularizer.mean()
        return loss_1tR, regularizer


class ContrastiveTask_11(FeatureMapContrastiveTask):

    def __init__(self, tclip=10.0):
        """
        This task compares two feature maps with the same dimensions. The feature maps are expected to be views
        from the same image

        .. code-block:: python

            # pseudocode
            phi = lambda a, b: mat_mul(a, b)

            loss11 = phi(f1, g1)

            total_loss = loss11

        To use this task, pass in two sets of feature maps

        Example::

            task = ContrastiveTask_11()

            # 1 feature map per image
            f1 = encoder(x_pos)
            g1 = encoder(x_anchor)

            # will only
            loss, regularizer = task(x1_maps=f1, x2_maps=g1)

        """
        super().__init__(tclip)

    def contrastive_task(self, x1_maps, x2_maps):
        x1_tgt = x1_maps
        x2_tgt = x2_maps

        if isinstance(x1_maps, (list, tuple)):
            x1_tgt = x1_maps[0]
            x2_tgt = x2_maps[0]

        # (b, dim, w. h)
        batch_size, emb_dim, h, w = x1_tgt.size()

        mask = self.masks[h]

        # -----------------
        # SOURCE VECTORS
        # 1 feature vector per image per feature map location
        # img1 -> img2
        r1_src_x1 = self._sample_src_ftr(x1_tgt, mask)
        r1_src_x2 = self._sample_src_ftr(x2_tgt, mask)

        # pick which map to use for negative samples
        x2_tgt = x2_maps[0]
        x1_tgt = x1_maps[0]

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size, device=r1_src_x1.device)

        # -----------------
        # NCE COSTS
        # compute costs for 1->5 prediction
        # use last layer to predict the layer with (5x5 features)
        loss_fwd, regularizer_fwd = self.nce_loss(r1_src_x1, x2_tgt, diag_mat)  # img 1
        loss_back, regularizer_back = self.nce_loss(r1_src_x2, x1_tgt, diag_mat)  # img 2

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss = 0.5 * (loss_fwd + loss_back)
        loss = loss.mean()

        regularizer = 0.5 * (regularizer_fwd + regularizer_back)
        regularizer = regularizer.mean()
        return loss, regularizer


def tanh_clip(x, clip_val=10.):
    """
    soft clip values to the range [-clip_val, +clip_val]
    """
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip
