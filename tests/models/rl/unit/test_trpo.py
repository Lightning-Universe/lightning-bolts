from pytorch_lightning import Trainer

from pl_bolts.models.rl.trpo_model import TRPO


def test_training_categorical():
    model = TRPO("CartPole-v0", batch_size=16)
    trainer = Trainer(max_epochs=1)
    trainer.fit(model)


def test_training_continous():
    model = TRPO("MountainCarContinuous-v0", batch_size=16)
    trainer = Trainer(max_epochs=1)
    trainer.fit(model)
