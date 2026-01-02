# -*- coding: utf-8 -*-
import copy
import functools
from typing import List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger as log
import os
import time
from typing import Any, Dict, Callable, Iterable, List, Mapping, Optional, Tuple, Union
from copy import deepcopy
from core.model.build import split_up_model, ResNetDomainNet126
from core.model.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_V2_MASK, IMAGENET_D109_MASK
from torch.nn.utils.weight_norm import WeightNorm

__all__ = ["setup"]

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def sample_selective_softplus_energy_alignment(input, ratio=0.5, temp=1.0):
    num_chunk = int(1/ratio)

    energy = -temp * torch.logsumexp(input / temp, dim=1)
    values, indices = energy.detach().sort()
    tar_energy = values.mean()

    src_energy_approx = torch.chunk(values, num_chunk)[0].mean()
    diff = energy - src_energy_approx
    loss = nn.Softplus()(diff)
    return loss
    
def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def weighted_lcs(input, cls_weight, thr=0.):
    classwise_virtual_logits = torch.matmul(cls_weight, cls_weight.T)
    #classwise_logit_dir = classwise_virtual_logits / classwise_virtual_logits.norm(dim=-1)

    # simple pseudo-labeling
    prob_all, indices_all = input.softmax(-1).max(-1)
    virtual_logits = classwise_virtual_logits[indices_all]
    conf_indices = prob_all>=thr
    loss = 1 - F.cosine_similarity(input, virtual_logits, dim=-1)
    loss = (prob_all.detach() - thr).exp() * loss
    loss = loss[conf_indices]

    return loss


class AEA(nn.Module):
    def __init__(self, cfg, model: nn.Module):
        super().__init__()
        model = configure_model(model)
        params, param_names = collect_params(model)
        self.cfg = cfg
        if cfg.OPTIM.METHOD == "SGD":
            optimizer = optim.SGD(
                params, 
                lr=float(cfg.OPTIM.LR),
                dampening=cfg.OPTIM.DAMPENING,
                momentum=float(cfg.OPTIM.MOMENTUM),
                weight_decay=float(cfg.OPTIM.WD),
                nesterov=cfg.OPTIM.NESTEROV
            )
        elif cfg.OPTIM.METHOD == "Adam":
            optimizer = optim.Adam(
                params, 
                lr=float(cfg.OPTIM.LR),
                betas=(cfg.OPTIM.BETA, 0.999),
                weight_decay=float(cfg.OPTIM.WD)
            )
        self.model = model
        self.optimizer = optimizer
        # loss weight
        self.lamb1 = cfg.ADAPTER.AEA.lamb1
        self.lamb2 = cfg.ADAPTER.AEA.lamb2
        self.lamb3 = cfg.ADAPTER.AEA.lamb3
        self.beta = cfg.ADAPTER.AEA.decay_beta
        self.lcs_thr = cfg.ADAPTER.AEA.lcs_thr

        self.filter_thr = math.log(1000) * 0.40
        self.ss_ratio = cfg.ADAPTER.AEA.ss_ratio  # sample selection ratio for energy alignment
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.feature_extractor, self.classifier = split_up_model(self.model, cfg.MODEL.ARCH, cfg.CORRUPTION.DATASET)
        if self.cfg.CORRUPTION.DATASET in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
            self.mask = eval(f"{cfg.CORRUPTION.DATASET.upper()}_MASK")
            self.cls_weight = self.classifier[0].weight[self.mask, :]
        else:
            self.mask = None
            self.cls_weight = self.classifier.weight

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, y=None, adapt=True):
        if not adapt:
            with torch.no_grad():
                outputs = self.model(x)
                return outputs

        loss, outputs = self.forward_and_adapt(x)

        return outputs

    def forward_and_adapt(self, x):
        self.optimizer.zero_grad()

        with torch.enable_grad():
            outputs = self.model(x)

            em_loss = softmax_entropy(outputs).mean(0)
            energy_loss = sample_selective_softplus_energy_alignment(outputs, ratio=self.ss_ratio)
            energy_loss = energy_loss.mean(0)
            logit_csim_loss = weighted_lcs(outputs, self.cls_weight, self.lcs_thr).mean(0)

            loss = em_loss \
                    + self.lamb2 * energy_loss \
                    + self.lamb3 * logit_csim_loss
            loss.backward()
        self.optimizer.step()

        return loss, outputs
    

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def setup(model, cfg):
    if isinstance(model, ResNetDomainNet126):  # https://github.com/pytorch/pytorch/issues/28594
        for module in model.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    delattr(module, hook.name)

    log.info("Setup TTA method: AEA")
    TTA_model = AEA(
        cfg,
        model
    )
    return TTA_model