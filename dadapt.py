# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.optim
import math
import logging
import numpy as np

class DAdaptSGD(torch.optim.Optimizer):
    def __init__(self, params,
        lr=1.0,
        momentum=0.0,
        weight_decay=0,
        log_every=0,
        d0=1e-6, growth_rate=float('inf')):

        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,
            momentum=momentum,
            weight_decay=weight_decay, k=0,
            log_every=log_every,
            numerator_weighted=0.0,
            d=d0,
            growth_rate=growth_rate)
        self.loggables = {}

        try:
            self.rank = torch.distributed.get_rank()
        except:
            self.rank = 0

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        lr = max(group['lr'] for group in self.param_groups)

        decay = group['weight_decay']
        momentum = group['momentum']
        log_every = group['log_every']
        ck = 1 - momentum
        k = group['k']

        numerator_weighted = group['numerator_weighted']
        growth_rate = group['growth_rate']
        d = group['d']

        group = self.param_groups[0]

        sk_sq = 0.0

        if k == 0:
            g_sq = 0.0
            for group in self.param_groups:
                group_lr = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data

                    # Apply weight decay
                    if decay != 0:
                        grad.add(p.data, alpha=decay)

                    state = self.state[p]

                    if group_lr > 0.0:
                        g_sq += (grad * grad).sum().item()

            global_gsq = g_sq
            group['g0_norm'] = g0_norm = math.sqrt(global_gsq)

        # G
        g0_norm = group['g0_norm']

        # lambda_k = ...
        dlr = d*lr/g0_norm

        for group in self.param_groups:
            group_lr = group['lr']
            if group_lr not in [lr, 0.0]:
                raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'z' not in state:
                    z = state['z'] = torch.clone(p.data).detach()
                    s = state['s'] = torch.zeros_like(p.data).detach()
                    x0 = state['x0'] = torch.clone(p.data).detach()

                # Apply weight decay
                if decay != 0:
                    grad.add_(p.data, alpha=decay)

                s = state['s']

                if group_lr > 0.0:
                    numerator_weighted += dlr * torch.dot(grad.flatten(), s.flatten()).item()

                    s.data.add_(grad, alpha=dlr)
                    sk_sq += (s * s).sum().item()
            ######

        d_hat = d

        if lr > 0.0:
            global_sk_sq = sk_sq
            global_numerator_weighted = numerator_weighted

            d_hat = 2*global_numerator_weighted/math.sqrt(global_sk_sq)
            d = max(d, min(d_hat, d*growth_rate))


        # if we have not done any updates
        # if we have any gradients available, will have sk_sq > 0 (unless \|g\|=0)
        if global_sk_sq == 0:
            return loss

        if log_every > 0 and k % log_every == 0:
            logging.info(f"(r={self.rank},k={k}) dlr: {dlr} d_hat: {d_hat}, d: {d}. sk_norm={math.sqrt(global_sk_sq)} numerator_weighted={global_numerator_weighted} g0_norm={g0_norm}")

        for group in self.param_groups:
            group['numerator_weighted'] = numerator_weighted
            group['d'] = d
            group['g0_norm'] = g0_norm
            ######################################
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                s = state['s']
                x0 = state['x0']
                z = state['z']

                # z step
                z.data.copy_(x0 - s)

                # x step
                p.data.mul_(1-ck).add_(z, alpha=ck)

            group['k'] = k + 1

        return loss


class DAdaptDual(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, log_every=0, d0=1e-6):

        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,
            k=0,
            log_every=log_every,
            d_numerator=0.0,
            s=0.0,
            gamma=0.0,
            d=d0)
        self.loggables = {}

        try:
            self.rank = torch.distributed.get_rank()
        except:
            self.rank = 0

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        k = group['k']

        # print(f"Param Groups in k = {k}: {self.param_groups}")
        # print(f"States: {self.state}")

        d_numerator = group['d_numerator']
        old_gamma = group['gamma']

        new_grad_norm = 0

        old_d = group['d']
        old_s_norm = 0
        new_s_norm = 0

        # updating gamma
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            state = self.state[p]

            if 's' not in state:
                s = state['s'] = torch.zeros_like(p.data).detach()
                x0 = state['x0'] = torch.clone(p.data).detach()

            s = state['s']
            old_s_norm += (s * s).sum().item()
            # print(f"Grad = {grad}")

            s.data.add_(grad, alpha=old_d)

            new_grad_norm += (grad * grad).sum().item()
            new_s_norm += (s * s).sum().item()


        if old_gamma > 0.0:
            # print(old_s_norm, "sadf")
            d_numerator += -1 * old_gamma * (old_d * old_d) * (grad * grad).sum().item() - old_gamma * old_s_norm
            new_gamma = 1 / np.sqrt(((1 / (old_gamma * old_gamma)) + new_grad_norm))
        else:
            new_gamma = math.sqrt(1 / new_grad_norm)
        d_numerator = d_numerator + new_gamma * new_s_norm

        d_hat = (d_numerator) / (2 * math.sqrt(new_s_norm))
        new_d = max(old_d, d_hat)

        if new_s_norm == 0:
            return loss

        for group in self.param_groups:
            group['d_numerator'] = d_numerator
            group['d'] = new_d
            group['gamma'] = new_gamma
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                s = state['s']
                x0 = state['x0']

                # print(f"new gamma = {new_gamma}, \tnew_s_norm = {new_s_norm}, \ts = {s}, \tdhat = {d_hat}, \td_numerator = {d_numerator}, \tnew x = {x0 - new_gamma * s}")

                p.data = x0 - new_gamma * s

            group['k'] = k + 1
            # print()

        return loss

