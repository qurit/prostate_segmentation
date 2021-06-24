# Original file from https://github.com/facebookresearch/detectron2/blob/master/detectron2/solver/lr_scheduler.py
# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List

import torch
from fvcore.common.config import CfgNode
from fvcore.common.param_scheduler import \
    MultiStepParamScheduler, LinearParamScheduler, CompositeParamScheduler, ParamScheduler


def build_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    # setup scheduler for decreasing lr at specified steps
    steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
    reduce_lr_sched = MultiStepParamScheduler(
        values=[cfg.SOLVER.GAMMA ** k for k in range(len(steps) + 1)],
        milestones=steps,
        num_updates=cfg.SOLVER.MAX_ITER,
    )

    # setup scheduler for warming up lr at start of run
    warmup_length = cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER
    warmup_lr_sched = LinearParamScheduler(
        start_value=cfg.SOLVER.WARMUP_FACTOR * reduce_lr_sched(0.0),
        end_value=reduce_lr_sched(warmup_length)  # the value to reach when warmup ends
    )

    scheds = CompositeParamScheduler(
        schedulers=[warmup_lr_sched, reduce_lr_sched],
        interval_scaling=["rescaled", "fixed"],
        lengths=[warmup_length, 1 - warmup_length]
    )

    return LRMultiplier(optimizer, multiplier=scheds, max_iter=cfg.SOLVER.MAX_ITER)


class LRMultiplier(torch.optim.lr_scheduler._LRScheduler):
    """
    Taken from detectron2/solver/lr_scheduler.py
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        multiplier: ParamScheduler,
        max_iter: int,
        last_iter: int = -1,
    ):
        """
        Args:
            optimizer, last_iter: See ``torch.optim.lr_scheduler._LRScheduler``.
                ``last_iter`` is the same as ``last_epoch``.
            multiplier: a fvcore ParamScheduler that defines the multiplier on
                every LR of the optimizer
            max_iter: the total number of training iterations
        """
        if not isinstance(multiplier, ParamScheduler):
            raise ValueError(
                "_LRMultiplier(multiplier=) must be an instance of fvcore "
                f"ParamScheduler. Got {multiplier} instead."
            )
        self._multiplier = multiplier
        self._max_iter = max_iter
        super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self):
        # fvcore schedulers are stateless. Only keep pytorch scheduler states
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def get_lr(self) -> List[float]:
        multiplier = self._multiplier(self.last_epoch / self._max_iter)
        return [base_lr * multiplier for base_lr in self.base_lrs]