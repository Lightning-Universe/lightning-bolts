import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plm
import torch
import torch.nn.functional as F

from pl_bolts.models.self_supervised import SSLEvaluator


class SSLFineTuner(pl.LightningModule):

    def __init__(self, backbone, in_features, num_classes, hidden_dim=1024):
        """
        Finetunes a self-supervised learning backbone using the standard evaluation protocol of a singler layer MLP
        with 1024 units

        Example::

            from pl_bolts.utils.self_supervised import SSLFineTuner
            from pl_bolts.models.self_supervised import CPCV2
            from pl_bolts.datamodules import CIFAR10DataModule
            from pl_bolts.models.self_supervised.cpc.transforms import CPCEvalTransformsCIFAR10,
                                                                        CPCTrainTransformsCIFAR10

            # pretrained model
            backbone = CPCV2.load_from_checkpoint(PATH, strict=False)

            # dataset + transforms
            dm = CIFAR10DataModule(data_dir='.')
            dm.train_transforms = CPCTrainTransformsCIFAR10()
            dm.val_transforms = CPCEvalTransformsCIFAR10()

            # finetuner
            finetuner = SSLFineTuner(backbone, in_features=backbone.z_dim, num_classes=backbone.num_classes)

            # train
            trainer = pl.Trainer()
            trainer.fit(finetuner, dm)

            # test
            trainer.test(datamodule=dm)

        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
            num_classes: classes of the dataset
            hidden_dim: dim of the MLP (1024 default used in self-supervised literature)
        """
        super().__init__()
        self.backbone = backbone
        self.ft_network = SSLEvaluator(
            n_input=in_features,
            n_classes=num_classes,
            p=0.2,
            n_hidden=hidden_dim
        )

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log_dict({'val_acc': acc, 'val_loss': loss}, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log_dict({'test_acc': acc, 'test_loss': loss})
        return loss

    def shared_step(self, batch):
        x, y = batch

        with torch.no_grad():
            feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        logits = self.ft_network(feats)
        loss = F.cross_entropy(logits, y)
        acc = plm.accuracy(logits, y)

        return loss, acc

    def configure_optimizers(
        self,
    ):
        return torch.optim.Adam(self.ft_network.parameters(), lr=0.0002)
