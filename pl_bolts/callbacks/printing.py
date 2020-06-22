from pytorch_lightning.callbacks import Callback
import copy
import pandas as pd


class PrintCallback(Callback):
    def __init__(self, columns=['val_loss']):
        """
        Prints a table with the metrics in columns on every epoch end
        """
        self.columns = columns
        self.metrics = []

    def on_epoch_end(self, trainer, pl_module):
        metrics_dict = copy.copy(trainer.callback_metrics)
        del metrics_dict['loss']

        self.metrics.append(metrics_dict)
        metrics_df = pd.DataFrame.from_records(self.metrics, columns=self.columns)
        # display(metrics_df)
