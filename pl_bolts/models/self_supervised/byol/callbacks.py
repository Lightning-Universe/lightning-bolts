import torch
import math
import pytorch_lightning as pl


class BYOLMAWeightUpdate(pl.Callback):

    def __init__(self, initial_tau=0.996):
        """
        Weight update rule from BYOL.
        Your model should have a:
            - self.online_network
            - self.target_network

        Updates the target_network params using an exponential moving average update rule weighted by tau.
        BYOL claims this keeps the online_network from collapsing.

        Arg:
            initial_tau: starting tau. Auto-updates with every training step
        """
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_batch_end(self, trainer, pl_module):
        # get networks
        online_net = pl_module.online_network
        target_net = pl_module.target_network

        # update weights
        self.update_weights(online_net, target_net)

        # update tau after
        self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module, trainer):
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / trainer.max_steps) + 1) / 2
        return tau

    def update_weights(self, online_net, target_net):
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(online_net.named_parameters(), target_net.named_parameters()):
            if 'weight' in name:
                target_p.data = self.current_tau * target_p.data + (1 - self.current_tau) * online_p.data
