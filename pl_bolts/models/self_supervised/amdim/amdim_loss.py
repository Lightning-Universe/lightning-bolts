import torch
from torch import nn
import numpy as np


class AMDIMLossNCE(nn.Module):
    '''
    Input is fixed as r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2.
    '''

    def __init__(self, tclip=10.):
        super().__init__()
        # construct masks for sampling source features from 5x5 layer
        # (b, 1, 5, 5)

        self.tclip = torch.tensor(tclip)
        self.masks_r5 = nn.Parameter(self.feat_size_w_mask(5), requires_grad=False)
        self.masks = {}

    def feat_size_w_mask(self, w):
        masks_r5 = np.zeros((w, w, 1, w, w))
        for i in range(w):
            for j in range(w):
                masks_r5[i, j, 0, i, j] = 1
        masks_r5 = torch.tensor(masks_r5).type(torch.uint8)
        masks_r5 = masks_r5.reshape(-1, 1, w, w)
        return masks_r5

    def nce_loss(self, r_src, r_trg, mask_mat):
        '''
        Compute the NCE scores for predicting r_src->r_trg.
        Input:
          r_src    : (batch_size, emb_dim)
          r_trg    : (emb_dim, n_batch * w* h) (ie: nb_feat_vectors x embedding_dim)
          mask_mat : (n_batch_gpu, n_batch)
        Output:
          raw_scores : (n_batch_gpu, n_locs)
          nce_scores : (n_batch_gpu, n_locs)
          lgt_reg    : scalar
        '''

        # RKHS = embedding dim
        batch_size, emb_dim = r_src.size()
        nb_feat_vectors = r_trg.size(1) // batch_size

        # (b, b) -> (b, b, nb_feat_vectors)
        # all zeros with ones in diagonal tensor... (ie: b1 b1 are all 1s, b1 b2 are all zeros)
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, nb_feat_vectors).float()

        # negative mask
        # one = torch.ones_like(mask_pos)
        mask_neg = 1. - mask_pos

        # -------------------------------
        # ALL SCORES COMPUTATION
        # compute src->trg raw scores for batch
        # (b, dim) x (dim, nb_feats*b) -> (b, b, nb_feats)
        # vector for each img in batch times all the vectors of all images in batch
        raw_scores = torch.mm(r_src, r_trg)
        raw_scores = raw_scores.reshape(batch_size, batch_size, nb_feat_vectors).float()

        # -----------------------
        # STABILITY TRICKS
        # trick 1: weighted regularization term
        raw_scores = raw_scores / emb_dim**0.5
        lgt_reg = 5e-2 * (raw_scores**2).mean()

        # trick 2: tanh clip
        raw_scores = tanh_clip(raw_scores, clip_val=self.tclip).float()

        '''
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        '''
        # ----------------------
        # EXTRACT POSITIVE SCORES
        # use the index mask to pull all the diagonals which are b1 x b1
        # (batch_size, nb_feat_vectors)
        pos_scores = (mask_pos * raw_scores).sum(dim=1).float()

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

        # FULL NCE
        # NUMERATOR
        # compute numerators for the NCE log-softmaxes
        pos_shiftexp = pos_scores - neg_maxes

        nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = -nce_scores.mean().float()

        return nce_scores, lgt_reg

    def _sample_src_ftr(self, r_cnv, masks):
        # get feature dimensions
        n_batch = r_cnv.size(0)
        feat_dim = r_cnv.size(1)

        if masks is not None:
            # subsample from conv-ish r_cnv to get a single vector
            mask_idx = torch.randint(0, masks.size(0), (n_batch,))
            r_cnv = torch.masked_select(r_cnv, masks[mask_idx]).float()

        # flatten features for use as globals in glb->lcl nce cost
        r_vec = r_cnv.reshape(n_batch, feat_dim)
        return r_vec

    def build_mask_cache(self, r5_x1):
        # cache masks
        batch_size, emb_dim, h, w = r5_x1.size()
        if len(self.masks) == 0:
            m1 = r5_x1

            # make mask
            if h not in self.masks:
                mask = self.feat_size_w_mask(h)
                mask = mask.cuda(m1.device.index)
                self.masks[h] = mask

        masks_r5 = self.masks[h]
        masks_r5 = masks_r5.cuda(r5_x1.device.index)

        return masks_r5

    def forward(self, r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2):
        '''
        Compute nce infomax costs for various combos of source/target layers.
        Compute costs in both directions, i.e. from/to both images (x1, x2).
        rK_x1 are features from source image x1.
        rK_x2 are features from source image x2.
        '''

        # masks_r5 = self.build_mask_cache(r5_x1)
        masks_r5 = self.masks_r5

        # (b, dim, w. h)
        batch_size, emb_dim, _, _ = r1_x1.size()

        # -----------------
        # SOURCE VECTORS
        # 1 feature vector per image per feature map location
        # img 1
        r1_src_1 = self._sample_src_ftr(r1_x1, None)
        r5_src_1 = self._sample_src_ftr(r5_x1, masks_r5)

        # img 2
        r1_src_2 = self._sample_src_ftr(r1_x2, None)
        r5_src_2 = self._sample_src_ftr(r5_x2, masks_r5)

        # -----------------
        # TARGET VECTORS
        # before shape: (n_batch, emb_dim, w, h)
        r5_trg_1 = r5_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r7_trg_1 = r7_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r5_trg_2 = r5_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r7_trg_2 = r7_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        # after shape: (emb_dim, n_batch * w * h)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_x1.device.index)

        # -----------------
        # NCE COSTS
        # compute costs for 1->5 prediction
        # use last layer to predict the layer with (5x5 features)
        loss_1t5_1, regularizer_1t5_1 = self.nce_loss(r1_src_1, r5_trg_2, diag_mat)  # img 1
        loss_1t5_2, regularizer_1t5_2 = self.nce_loss(r1_src_2, r5_trg_1, diag_mat)  # img 2

        # compute costs for 1->7 prediction
        # use last layer to predict the layer with (7x7 features)
        loss_1t7_1, regularizer_1t7_1 = self.nce_loss(r1_src_1, r7_trg_2, diag_mat)  # img 1
        loss_1t7_2, regularizer_1t7_2 = self.nce_loss(r1_src_2, r7_trg_1, diag_mat)  # img 2

        # compute costs for 5->5 prediction
        # use (5x5) layer to predict the (5x5) layer
        loss_5t5_1, regularizer_5t5_1 = self.nce_loss(r5_src_1, r5_trg_2, diag_mat)  # img 1
        loss_5t5_2, regularizer_5t5_2 = self.nce_loss(r5_src_2, r5_trg_1, diag_mat)  # img 2

        # combine costs for optimization
        loss_1t5 = 0.5 * (loss_1t5_1 + loss_1t5_2)
        loss_1t7 = 0.5 * (loss_1t7_1 + loss_1t7_2)
        loss_5t5 = 0.5 * (loss_5t5_1 + loss_5t5_2)

        # regularizer
        regularizer = 0.5 * (regularizer_1t5_1 + regularizer_1t5_2 +
                             regularizer_1t7_1 + regularizer_1t7_2 +
                             regularizer_5t5_1 + regularizer_5t5_2)

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss_1t5 = loss_1t5.mean()
        loss_1t7 = loss_1t7.mean()
        loss_5t5 = loss_5t5.mean()
        regularizer = regularizer.mean()
        return loss_1t5, loss_1t7, loss_5t5, regularizer


def tanh_clip(x, clip_val=10.):
    '''
    soft clip values to the range [-clip_val, +clip_val]
    '''
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip