"""
This is just the layout for FasterRCNN using lightning
"""
from argparse import ArgumentParser

import torch
import torchvision
from pytorch_lightning import LightningModule, Trainer
from test_tube import HyperOptArgumentParser
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from pl_bolts.models.autoencoders import BasicAE


class BBFasterRCNN(LightningModule):

    def __init__(self, pretrained_path):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.output_dim = 800 * 800

        # ------------------
        # PRE-TRAINED MODEL
        # ------------------
        ae = BasicAE.load_from_checkpoint(pretrained_path)
        ae.freeze()

        self.backbone = ae.encoder
        self.backbone.c3_only = True
        self.backbone.out_channels = 32

        # ------------------
        # FAST RCNN
        # ------------------
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2)
        self.fast_rcnn = FasterRCNN(
            self.backbone,
            num_classes=9,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        # for unfreezing encoder later
        self.frozen = True

    def forward(self, ssr, targets):
        losses_dict = self.fast_rcnn(ssr, targets)
        return losses_dict

    def _run_step(self, batch, batch_idx, step_name):
        # images, target and roadimage are tuples of length batchsize
        # bb coords in target using (-40, 40) scale
        images, raw_target, road_image = batch

        # adjust format for FastRCNN
        # images: list([3, 800, 800]); target: list( {'boxes': [N, 4], 'labels': [N] } )
        # here bb coords are already transformed to (0, 800) range
        images, target = self._format_for_fastrcnn(images, raw_target)

        # run a forward step
        # depending on whether its a train or validation step - losses is dict with different keys
        losses = self(images, target)

        # in training, the output is a dict of scalars
        if step_name == 'train':
            loss_classifier = losses['loss_classifier'].double()
            loss_box_reg = losses['loss_box_reg'].double()
            loss_objectness = losses['loss_objectness'].double()
            loss_rpn_box_reg = losses['loss_rpn_box_reg'].double()
            loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg
            return loss, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
        else:
            # in val, the output is a dic of boxes and losses
            # ----------------------
            # LOG VALIDATION IMAGES
            # ----------------------
            if batch_idx % self.hparams.output_img_freq == 0:
                # --- log one validation predicted image ---
                # [N, 4] - seems to be [100,4]
                predicted_coords_0 = losses[0]['boxes']
                # transform [N, 4] -> [N, 2, 4]
                # note these predictions are on [0, 800] scale
                predicted_coords_0 = self._change_to_old_coord_sys(predicted_coords_0)
                pred_categories_0 = losses[0]['labels']  # [N]

                target_coords_0 = raw_target[0]['bounding_box'] * 10 + 400
                # [N, 4] -> [N, 2, 4]
                # target_coords_0 = self._change_to_old_coord_sys(target_coords_0)
                target_categories_0 = raw_target[0]['category']

    def _format_for_fastrcnn(self, images, target):
        # split batch into list of single images
        # [b, 3, 256, 1836] --> list of length b with elements [3, 256, 1836]
        images = list(image.float() for image in images)

        target = [{k: v for k, v in t.items()} for t in target]
        for d in target:
            d['boxes'] = d.pop('bounding_box')
            d['labels'] = d.pop('category')

            # Change coords to (x0, y0, x1, y1) ie top left and bottom right corners
            # TODO: verify
            d['boxes'] = self._change_coord_sys(d['boxes']).float()
            # num_boxes = d['boxes'].size(0)
            # d['boxes'] = d['boxes'][:, :, [0, -1]].reshape(num_boxes, -1).float()

        return images, target

    def training_step(self, batch, batch_idx):

        # unfreeze after so many epochs
        if self.current_epoch >= self.hparams.unfreeze_epoch_no and self.frozen:
            self.frozen = False
            self.backbone.train()

        train_loss, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg = \
            self._run_step(batch, batch_idx, step_name='train')
        train_tensorboard_logs = {'train_loss': train_loss,
                                  'train_loss_classifier': loss_classifier,
                                  'train_loss_box_reg': loss_box_reg,
                                  'train_loss_objectness': loss_objectness,
                                  'train_loss_rpn_box_reg': loss_rpn_box_reg}

        return {'loss': train_loss, 'log': train_tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        self._run_step(batch, batch_idx, step_name='valid')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser], add_help=False)

        # want to optimize this parameter
        # parser.opt_list('--batch_size', type=int, default=16, options=[16, 10, 8], tunable=False)
        parser.opt_list('--learning_rate', type=float, default=0.001, options=[1e-3, 1e-4, 1e-5], tunable=True)
        parser.add_argument('--batch_size', type=int, default=10)
        # fixed arguments
        parser.add_argument('--output_img_freq', type=int, default=100)
        parser.add_argument('--unfreeze_epoch_no', type=int, default=0)

        parser.add_argument('--mse_loss', default=False, action='store_true')
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BBFasterRCNN.add_model_specific_args(parser)
    args = parser.parse_args()

    model = BBFasterRCNN(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)
