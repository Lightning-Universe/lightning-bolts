from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import distributed as dist


class SWAVLoss(nn.Module):
    def __init__(
        self,
        temperature: float,
        crops_for_assign: tuple,
        nmb_crops: tuple,
        sinkhorn_iterations: int,
        epsilon: float,
        gpus: int,
        num_nodes: int,
    ):
        """Implementation for SWAV loss function.

        Args:
            temperature:  loss temperature
            crops_for_assign: list of crop ids for computing assignment
            nmb_crops: number of global and local crops, ex: [2, 6]
            sinkhorn_iterations: iterations for sinkhorn normalization
            epsilon: epsilon val for swav assignments
            gpus: number of gpus per node used in training, passed to SwAV module
                to manage the queue and select distributed sinkhorn
            num_nodes:  num_nodes: number of nodes to train on
        """
        super().__init__()
        self.temperature = temperature
        self.crops_for_assign = crops_for_assign
        self.softmax = nn.Softmax(dim=1)
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.nmb_crops = nmb_crops
        self.gpus = gpus
        self.num_nodes = num_nodes
        if self.gpus * self.num_nodes > 1:
            self.assignment_fn = self.distributed_sinkhorn
        else:
            self.assignment_fn = self.sinkhorn

    def forward(
        self,
        output: torch.Tensor,
        embedding: torch.Tensor,
        prototype_weights: torch.Tensor,
        batch_size: int,
        queue: Optional[torch.Tensor] = None,
        use_queue: bool = False,
    ) -> Tuple[int, Optional[torch.Tensor], bool]:
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[batch_size * crop_id : batch_size * (crop_id + 1)]

                # Time to use the queue
                if queue is not None:
                    if use_queue or not torch.all(queue[i, -1, :] == 0):
                        use_queue = True
                        out = torch.cat((torch.mm(queue[i], prototype_weights.t()), out))
                    # fill the queue
                    queue[i, batch_size:] = self.queue[i, :-batch_size].clone()  # type: ignore
                    queue[i, :batch_size] = embedding[crop_id * batch_size : (crop_id + 1) * batch_size]
                # get assignments
                q = torch.exp(out / self.epsilon).t()
                q = self.assignment_fn(q, self.sinkhorn_iterations)[-batch_size:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                p = self.softmax(output[batch_size * v : batch_size * (v + 1)] / self.temperature)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)  # type: ignore
        return loss, queue, use_queue

    def sinkhorn(self, Q: torch.Tensor, nmb_iters: int) -> torch.Tensor:
        """Implementation of Sinkhorn clustering."""
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)

                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q: torch.Tensor, nmb_iters: int) -> torch.Tensor:
        """Implementation of Distributed Sinkhorn."""
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for _ in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
