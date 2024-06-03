"""
PyTorch implementation of the DoG/LDoG optimizers (Ivgi et al., 2023)
"""
import logging
from typing import Optional

import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class DualAveragingV2(Optimizer):
    def __init__(self, params, reps_rel: float = 1e-6, lr: float = 1.0,
                 weight_decay: float = 0.0, eps: float = 1e-8, init_beta: Optional[float] = None, c: float = 4):

        if lr <= 0.0:
            raise ValueError(f'Invalid learning rate ({lr}). Suggested value is 1.')
        if lr != 1.0:
            logger.warning(f'We do not recommend changing the lr parameter from its default value of 1')
        if init_beta is not None:
            if init_beta <= 0:
                raise ValueError(f'Invalid value for init_eta ({init_beta})')
            logger.info(f'Ignoring reps_rel since will be explicitly set init_eta to be {init_beta} (first step size)')
            reps_rel = 0
        else:
            if reps_rel <= 0.0:
                raise ValueError(f'Invalid reps_rel value ({reps_rel}). Suggested value is 1e-6 '
                                 '(unless the model uses batch-normalization, in which case suggested value is 1e-4)')

        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')

        self._first_step = True

        defaults = dict(reps_rel=reps_rel, lr=lr, weight_decay=weight_decay, eps=eps, init_beta=init_beta)
        self.c = c
        super(DualAveragingV2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DualAveragingV2, self).__setstate__(state)

    def state_dict(self) -> dict:
        state_dict = super(DualAveragingV2, self).state_dict()
        logger.info('retrieving DA state dict')
        state_dict['state']['_first_step'] = self._first_step
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        super(DoG, self).load_state_dict(state_dict)
        self._first_step = state_dict['state']['_first_step']
        logger.info(f'loaded DoG state dict')
        cuda = self.param_groups[0]['params'][0].device
        for group in self.param_groups:
            cuda_buffers = {'init_buffer'}
            for tgroup in group.keys():
                # this can cast all the tensors to the device. However, as it turns out,
                # we need ONLY the init_buffer to be on the params' device
                if tgroup != 'params':
                    device = cuda if tgroup in cuda_buffers else 'cpu'
                    if isinstance(group[tgroup], list) and len(group[tgroup]) > 0 and \
                            isinstance(group[tgroup][0], torch.Tensor):
                        group[tgroup] = [i.to(device) for i in group[tgroup]]
                    elif isinstance(group[tgroup], torch.Tensor):
                        group[tgroup] = group[tgroup].to(device)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        first_step = self._first_step

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            if first_step:
                init = group['init_buffer'] = [torch.clone(p).detach() for p in group['params']]
            else:
                init = group['init_buffer']

            if weight_decay > 0:
                for p in group['params']:
                    p.grad.add_(p, alpha=weight_decay)

            self._update_group_state(group, init)
            # self._override_init_eta_if_needed(group)

            for p, eta in zip(group['params'], group['eta']):
                if p.grad is None:
                    continue
                else:
                    p.data = group['x0'] - eta

        self._first_step = False

        return loss

    def _update_group_state(self, group, init):
        # treat all layers as one long vector
        if self._first_step:
            group['x0'] = torch.stack([torch.clone(p.data).detach() for p in group['params']])[0]
            group['rbar'] = group['reps_rel'] * (1 + torch.stack([p.norm() for p in group['params']]).norm())
            group['G'] = torch.stack([(p.grad.detach() ** 2).sum() for p in group['params']]).sum() + group['eps']
            group['s'] = torch.stack([(group['reps_rel'] * p.grad.detach()) for p in group['params']])[0]
        else:
            curr_d = torch.stack([torch.norm(p.detach() - pi) for p, pi in zip(group['params'], init)]).norm()
            group['rbar'] = torch.maximum(group['rbar'], curr_d)
            group['G'] += torch.stack([(p.grad.detach() ** 2).sum() for p in group['params']]).sum()
            group['s'] += torch.stack([(curr_d * p.grad.detach()) for p in group['params']])[0]
        assert group['G'] > 0, \
            f'DoG cannot work when G is not strictly positive. got: {group["G"]}'
        group['eta'] = [(1 / torch.sqrt(group['G'])) * group['s']] * len(group['params'])
        # print(group['s'])

    def _override_init_eta_if_needed(self, group):
        # Override init_eta if needed
        if self._first_step and group['init_eta'] is not None:
            init_eta = group['init_eta']
            logger.info(f'Explicitly setting init_eta value to {init_eta}')
            group['eta'] = [eta * 0 + init_eta for eta in group['eta']]
