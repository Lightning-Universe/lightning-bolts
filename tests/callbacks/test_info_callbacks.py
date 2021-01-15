from pl_bolts.callbacks import PrintTableMetricsCallback


def test_printtable_metrics_callback():
    callback = PrintTableMetricsCallback()

    metrics_a = {'loss': 1.0, 'epoch': 0}
    metrics_b = {'loss': 0.5, 'epoch': 2}

    class FakeTrainer(object):

        def __init__(self):
            self.callback_metrics = {}

    fake_trainer = FakeTrainer()
    fake_trainer.callback_metrics = metrics_a
    callback.on_epoch_end(fake_trainer, None)
    fake_trainer.callback_metrics = metrics_b
    callback.on_epoch_end(fake_trainer, None)

    assert len(callback.metrics) == 2
    assert callback.metrics[0] == metrics_a
    assert callback.metrics[1] == metrics_b
