import pytorch_lightning as pl
import torch

from pl_bolts.utils.warnings import warn_missing_pkg

try:
    from torchvision.models import resnet34
except ModuleNotFoundError:
    warn_missing_pkg('torchvision')  # pragma: no-cover

class ResNet34(pl.LightningModule):
    def __init__(self, 
                 learning_rate: float =2e-5, 
                 num_classes: int=50):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = resnet34(pretrained=True)
        self.model.fc = torch.nn.Linear(512, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_acc = pl.metrics.Accuracy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(self, batch, _):
        loss, *_ = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        loss, logits, y = self.step(batch)
        self.val_acc(logits, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)

        @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--num_classes", type=int, default=50)
        return parser

def run_cli():

    from pl_bolts.datamodules import ESC50DataModule

    pl.seed_everything(42)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ResNet34.add_model_specific_args(parser)
    args = parser.parse_args()

    datamodule = ESC50DataModule(args.data_dir)
    args.num_classes = datamodule.num_classes

    model = ResNet34(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    run_cli()