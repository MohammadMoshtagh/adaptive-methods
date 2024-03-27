import torch
from torch.optim import Optimizer
from typing import List, Optional, Any, Dict

class SGD(Optimizer):

    def __init__(self, model_params, lr: float = 1.0, weight_decay: float = 0):
        optimizer_params = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(model_params, optimizer_params)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)

    def _init_group(self, group, params_with_grad, d_p_list):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grads = []
            d_p_list = []

            self._init_group(group, params_with_grads, d_p_list)

            self.sgd(params_with_grads, d_p_list, group['lr'], group['weight_decay'])

        return loss

    def sgd(self, params, d_p_list, lr, weight_decay):
        for i, param in enumerate(params):
            d_p = d_p_list[i]
            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            param.data.add_(d_p, alpha=-lr)