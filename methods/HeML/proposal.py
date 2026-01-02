import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import torchvision.transforms as T 

from copy import deepcopy
from loguru import logger as log
from collections import defaultdict
from core.model.build import split_up_model
from methods.DCF.MCL import Prototype
from methods.DCF.transformers_cotta import get_tta_transforms
import torchvision
from einops import rearrange
from core.model.build import split_up_model, ResNetDomainNet126
from torch.nn.utils.weight_norm import WeightNorm


__all__ = ["setup"]

# +++ START: Helper functions and classes +++
def to_float(x):
    """Helper function to convert tensor to float32."""
    return x.float()

def get_low_pass_augmentation(kernel_size=9, sigma=(2.0, 4.0)):
    """
    Create low-pass filter transform (Gaussian Blur).
    """
    blur_transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return blur_transform

class GradientEdgeImage(nn.Module):
    def __init__(self, return_rgb_shape=True):
        """
        Args:
            return_rgb_shape: If True (default), returns (B, 3, H, W).
        """
        super().__init__()
        self.return_rgb_shape = return_rgb_shape
        
        # Define standard Sobel kernels
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        
        # Convert to Tensor
        self.register_buffer('weight_x', torch.tensor(kernel_x).view(1, 1, 3, 3))
        self.register_buffer('weight_y', torch.tensor(kernel_y).view(1, 1, 3, 3))

    def rgb_to_grayscale(self, x):
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        return gray

    def normalize_batch(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1) 
        min_val = x_flat.min(dim=2, keepdim=True)[0].view(B, C, 1, 1)
        max_val = x_flat.max(dim=2, keepdim=True)[0].view(B, C, 1, 1)
        eps = 1e-8
        normalized_x = (x - min_val) / (max_val - min_val + eps)
        return normalized_x

    def forward(self, x):
        if x.shape[1] == 3:
            x_gray = self.rgb_to_grayscale(x)
        else:
            x_gray = x

        grad_x = F.conv2d(x_gray, self.weight_x, padding=1)
        grad_y = F.conv2d(x_gray, self.weight_y, padding=1)
        
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        edge_image = self.normalize_batch(gradient_magnitude)
        
        if self.return_rgb_shape and x.shape[1] == 3:
            return edge_image.repeat(1, 3, 1, 1)
        
        return edge_image

class LayerWiseGramCollector(nn.Module):
    """
    Collects layer-wise Gram matrices.
    """
    def __init__(self, model, device='cuda', gram_dtype=torch.float32):
        super().__init__()
        self.model = model
        self.device = device
        self.gram_dtype = gram_dtype
        self.adaptable_layers = self._get_adaptable_layers()
        self.all_grams = {}
        self.all_hooks = []
        
    def _get_adaptable_layers(self):
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                layers.append({
                    'name': name,
                    'module': module,
                    'params': {f"{name}.{pn}": p for pn, p in module.named_parameters() 
                              if pn in ['weight', 'bias'] and p.requires_grad}
                })
        return layers
    
    def _create_hook_handler(self, layer_idx):
        def hook_handler(module, inputs, outputs):
            x = inputs[0].detach().to(self.gram_dtype)
            if len(x.shape) == 4:  # Conv layers
                x = x.permute(0, 2, 3, 1).contiguous()
                x = x.view(-1, x.size(-1))
            elif len(x.shape) == 3: # Transformer layers
                x = x.view(-1, x.size(-1))
            
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
            gram = torch.matmul(x.T, x) / x.shape[0]
            
            if layer_idx not in self.all_grams:
                self.all_grams[layer_idx] = gram
            else:
                self.all_grams[layer_idx] += gram
        
        return hook_handler
    
    def collect_all_grams_once(self, x):
        self.all_grams = {}
        self.clear_all_hooks()
        
        for idx, layer_info in enumerate(self.adaptable_layers):
            hook = layer_info['module'].register_forward_hook(
                self._create_hook_handler(idx)
            )
            self.all_hooks.append(hook)
        
        with torch.no_grad():
            _ = self.model(x)
        
        self.clear_all_hooks()
        return self.all_grams
    
    def clear_all_hooks(self):
        for hook in self.all_hooks:
            hook.remove()
        self.all_hooks = []

def gaussian_lowpass_mask(h, w, sigma_f, device, dtype):
    fy = torch.fft.fftfreq(h, d=1.0, device=device).reshape(h, 1)
    fx = torch.fft.fftfreq(w, d=1.0, device=device).reshape(1, w)
    f2 = fy**2 + fx**2
    mask = torch.exp(-0.5 * f2 / (sigma_f**2 + 1e-12))
    return mask.to(dtype=dtype)

def lowpass_filter(x, sigma_f=0.1):
    B, C, H, W = x.shape
    X = torch.fft.fft2(x, dim=(-2, -1))
    M = gaussian_lowpass_mask(H, W, sigma_f, x.device, X.dtype).view(1,1,H,W)
    Xf = X * M
    x_lp = torch.fft.ifft2(Xf, dim=(-2, -1)).real
    return x_lp

@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, update_all=False):
    if momentum < 1.0:
        for param_to_update, param_to_merge in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.cuda()
    return model_to_update

def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat

def softmax_clamp(logits: torch.Tensor) -> torch.Tensor:
    logits = F.softmax(logits, dim=1)
    logits = torch.clamp(logits, min=0.0, max=0.99)
    return logits

class SupSoftLikelihoodRatio(nn.Module):
    def __init__(self, gamma=1e-5):
        super(SupSoftLikelihoodRatio, self).__init__()
        self.gamma = gamma
        self.eps=1e-5

    def __call__(self, logits, target_logits):
        logits = softmax_clamp(logits)
        target_logits = softmax_clamp(target_logits)
        return - (logits * torch.log(
            (target_logits  * (1- self.gamma)) / ((1 - target_logits) + self.eps)  + self.gamma
        ) / (1- self.gamma)).sum(1)

def copy_model(model):
    if isinstance(model, ResNetDomainNet126):
        for module in model.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    if hasattr(module, hook.name):
                        delattr(module, hook.name)
        coppied_model = deepcopy(model)
        for module in model.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    hook(module, None)
    else:
        coppied_model = deepcopy(model)
    return coppied_model
    
@torch.jit.script
def consistency(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * y.log_softmax(1)).sum(1)

@torch.jit.script
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)

def copy_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer):
    model_state = deepcopy(model.state_dict())
    src_model = copy_model(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    for param in src_model.parameters():
        param.detach_()
    return model_state, optimizer_state, src_model

def load_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer,
                             model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model: nn.Module, cfg):
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            module.requires_grad_(True)
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            module.requires_grad_(True)
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.requires_grad_(True)
    return model

def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if 'layer4' in nm or 'blocks.9' in nm or 'blocks.10' in nm or 'blocks.11' in nm or 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names
# +++ END: Helper functions +++


class HeML(nn.Module):
    """Hybrid Empirical Model Learning (HeML) with Adaptive Bayesian Filtering update."""

    def __init__(self, cfg, model: nn.Module):
        super().__init__()
        params, param_names = collect_params(model)
        self.params = params
        
        lr = cfg.OPTIM.LR
        # Adjust learning rate based on arch and batch size
        if cfg.MODEL.ARCH == "resnet50_gn" or cfg.MODEL.ARCH == "resnet50":
            lr = (cfg.OPTIM.LR / 64) * cfg.TEST.BATCH_SIZE * 2 if cfg.TEST.BATCH_SIZE < 32 else cfg.OPTIM.LR
        elif cfg.MODEL.ARCH == "vit_b_16":
            lr = (0.001 / 64) * cfg.TEST.BATCH_SIZE
        if cfg.TEST.BATCH_SIZE == 1:
            lr = 2 * lr

        if cfg.OPTIM.METHOD == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=float(cfg.OPTIM.MOMENTUM), weight_decay=float(cfg.OPTIM.WD))
        elif cfg.OPTIM.METHOD == "Adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=float(cfg.OPTIM.WD))
        else:
            raise ValueError(f"Invalid optimizer method: {cfg.OPTIM.METHOD}")

        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "HeML requires >= 1 step(s) to forward and update"
                        
        self.model_state, self.optimizer_state, self.src_model = copy_model_and_optimizer(self.model, self.optimizer)
        self.hidden_model = copy_model(self.model)
        self.hidden_model.train()

        self.ema_model = copy_model(self.model)
        self.ema_model.train()
        
        # Use HeML configuration path
        self.MU = self.cfg.ADAPTER.HeML.MU
        self.EMA_TRIGGER = False
        self.sup_slr = SupSoftLikelihoodRatio(0.001)

        self.eps = 1e-6
        self.batch_index = 0
        self.batch_size = cfg.TEST.BATCH_SIZE
        self.entropy_margin = math.log(self.cfg.CORRUPTION.NUM_CLASS) * 0.65  
        self.feature_extractor, self.classifier = split_up_model(
            self.model, cfg.MODEL.ARCH, cfg.CORRUPTION.DATASET
        )
        self.sigma_f = self.cfg.ADAPTER.HeML.SIGMA_F

        self.current_model_probs = None
        self.transforms = get_tta_transforms(cfg=cfg, padding_mode="reflect", cotta_augs=True)
        self.strong_transforms = get_low_pass_augmentation(kernel_size=9, sigma=(2.0, 4.0))
        
        # [NEW] Edge Image Extractor
        self.edge_extractor = GradientEdgeImage(return_rgb_shape=True)
        
        # Initial Prototype Setup
        self.reset_prototypes() 

        self.param_group_names = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]

        # --- Bayesian Filtering Initialization (HeML Path) ---
        self.alpha = getattr(self.cfg.ADAPTER.HeML, "ALPHA", 0.9)
        self.q = getattr(self.cfg.ADAPTER.HeML, "Q", 1e-4)
        self.gamma = getattr(self.cfg.ADAPTER.HeML, "GAMMA", 0.9)
        self.post_type = getattr(self.cfg.ADAPTER.HeML, "TYPE", "op") 
        self.alpha_regmean = 0.0  
        self.normalize_grams = True 

        self.hidden_var = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gram_dtype = torch.float32 
        self.gram_collector = LayerWiseGramCollector(
            model=self.model,
            device=self.device,
            gram_dtype=self.gram_dtype
        )
        self.adaptable_layers = self.gram_collector.adaptable_layers
        
        # Momentum Grams variables
        self.historical_grams = {}
        self.gram_momentum = 0.4 

        # [Innovation] Gram-Guided Restoration Probability
        # Use HeML gram_momentum as base restoration probability
        self.restore_base_prob = getattr(self.cfg.ADAPTER.HeML, "gram_momentum", 0.1)

    def reset_prototypes(self):
        """Helper to reset the prototype memory bank."""
        self.prototype = Prototype(
            C=self.cfg.CORRUPTION.NUM_CLASS,
            dim= 256 if self.cfg.CORRUPTION.DATASET == "domainnet126" else self.cfg.MODEL.PROJECTION.EMB_DIM,
        )

    def forward(self, x, y=None, adapt=True):
        if adapt:
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)
        else:
            with torch.no_grad():
                outputs = self.model(x)
        self.batch_index += 1
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        
        # --- Loss calculation ---
        total_loss = 0
        
        # 1. Augmentations
        x_aug = self.transforms(x)
        x_strong = self.strong_transforms(x_aug)
        
        # 2. Edge Images
        x_edge = self.edge_extractor(x)
        
        # 3. Forward Passes
        strong_feats = self.feature_extractor(x_strong)
        x_aug = lowpass_filter(x_aug, sigma_f=self.sigma_f)
        weak_feats = self.feature_extractor(x_aug)
        orig_feats = self.feature_extractor(x)
        edge_feats = self.feature_extractor(x_edge) 
        
        weak_logits = self.classifier(weak_feats)
        strong_logits = self.classifier(strong_feats)
        orig_logits = self.classifier(orig_feats)
        edge_logits = self.classifier(edge_feats) 
    
        with torch.no_grad():
            _logits_A, _logits_B = orig_logits, strong_logits
            _logits_C = edge_logits 
            
            # Tier 1: Entropy Filtering
            entropy = softmax_entropy(_logits_A.detach())
            _entropy_idx = torch.where(entropy < self.entropy_margin)[0]
            
            prob_outputs = _logits_A[_entropy_idx].softmax(1)
            prob_outputs_strong = _logits_B[_entropy_idx].softmax(1)
            prob_outputs_edge = _logits_C[_entropy_idx].softmax(1) 
            
            cls1 = prob_outputs.argmax(dim=1)
            
            # Tier 2: Strong Augmentation Consistency (PCS)
            pcs = torch.gather(prob_outputs, 1, cls1.view(-1, 1)) - torch.gather(prob_outputs_strong, 1, cls1.view(-1, 1))
            pcs = pcs.view(-1).abs()
            _pcs_idx = torch.where(pcs > 0.01 )[0]

            # Tier 3: Gradient Edge Consistency (PCS - Edge)
            pcs_edge = torch.gather(prob_outputs, 1, cls1.view(-1, 1)) - torch.gather(prob_outputs_edge, 1, cls1.view(-1, 1))
            pcs_edge = pcs_edge.view(-1).abs()
            _edge_pcs_idx = torch.where(pcs_edge > 0.01)[0] 
            
            # Intersect Masks
            mask_len = len(_entropy_idx)
            mask_strong = torch.zeros(mask_len, dtype=torch.bool, device=self.device)
            mask_edge = torch.zeros(mask_len, dtype=torch.bool, device=self.device)
            
            mask_strong[_pcs_idx] = True
            mask_edge[_edge_pcs_idx] = True
            
            combined_mask = mask_strong & mask_edge
            _combined_idx = torch.where(combined_mask)[0]
            
            # Final mapping to batch dimension
            _fillter_idx = _entropy_idx[_combined_idx]
            
            mask = torch.zeros_like(entropy, dtype=torch.bool)
            mask[_fillter_idx] = True
            
        # Weight Calculation
        pcs_weights = self.get_pcs_weights(_logits_B, _logits_A)
        # Use HeML configuration items uniformly
        weights = torch.exp(pcs_weights) ** self.cfg.ADAPTER.HeML.PCS_W
        
        # Loss calculation
        slr_loss = ((
            self.sup_slr(orig_logits, orig_logits)[mask] * weights[mask] +
            self.sup_slr(orig_logits, weak_logits)[mask] * weights[mask] +
            self.sup_slr(orig_logits, strong_logits)[mask] * weights[mask] 
        )).mean(0)
        total_loss += slr_loss 
        
        if len(_fillter_idx) > 0:
            total_loss.backward()
            self.optimizer.step()

        # --- Adaptive Bayesian Filtering ---
        self.bayesian_filtering(x, logits=orig_logits)
        
        return orig_logits

    def _map_importance_to_params(self, importance, param_shape, device):
        if len(param_shape) == 1: 
            return importance if param_shape[0] == len(importance) else torch.ones(param_shape[0], device=device) * 0.5
        elif len(param_shape) == 2:  
            if param_shape[0] == len(importance): return importance.unsqueeze(1).expand(param_shape)
            elif param_shape[1] == len(importance): return importance.unsqueeze(0).expand(param_shape)
            else: return torch.ones(param_shape, device=device) * 0.5
        else:
            if param_shape[0] == len(importance):
                weights = importance.view(-1, *([1] * (len(param_shape) - 1)))
                return weights.expand(param_shape)
            else: return torch.ones(param_shape, device=device) * 0.5

    def compute_merge_weights_regmean(self, gram, param_shape, layer_idx):
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        regularized_gram = gram * (eye + (1 - eye) * self.alpha_regmean)
        importance = 1.0 / (torch.diag(regularized_gram) + 1e-8) if self.alpha_regmean != 0.0 else torch.diag(regularized_gram)
        
        num_layers = len(self.adaptable_layers)
        temperature = 2.0 - (layer_idx / (num_layers -1)) * 1.0 
        
        if temperature != 1.0:
            importance = torch.pow(importance, 1.0 / temperature)

        if self.normalize_grams and importance.sum() > 0:
            importance = importance / (importance.sum() + 1e-8)
            
        return self._map_importance_to_params(importance, param_shape, gram.device)

    @torch.no_grad()
    def _gram_weighted_merge(self, model_to_update, model_to_merge, all_grams, base_momentum, enable_restoration=False):
        """
        Gram-Guided Stochastic Layer-wise Merge for HeML.
        """
        merge_state_dict = model_to_merge.state_dict()
        for layer_idx, layer_info in enumerate(self.adaptable_layers):
            gram = all_grams.get(layer_idx, None)
            if gram is None: continue
            
            gram = gram.to(self.device)
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            regularized_gram = gram * (eye + (1 - eye) * self.alpha_regmean)
            raw_importance = 1.0 / (torch.diag(regularized_gram) + 1e-8)
            norm_importance = (raw_importance - raw_importance.min()) / (raw_importance.max() - raw_importance.min() + 1e-8)

            for param_name, param in layer_info['params'].items():
                if param_name not in merge_state_dict: continue
                
                merge_param_data = merge_state_dict[param_name].to(param.device)
                gram_weights = self.compute_merge_weights_regmean(gram, param.shape, layer_idx).to(param.device)
                adaptive_ratio = gram_weights * (1.0 - base_momentum)
                
                if enable_restoration and self.restore_base_prob > 0:
                    param_norm_imp = self._map_importance_to_params(norm_importance, param.shape, param.device)
                    prob_map = self.restore_base_prob * (1.0 - param_norm_imp) + 1e-6
                    restore_mask = torch.bernoulli(torch.clamp(prob_map, 0, 1)).float()
                    adaptive_ratio = torch.max(adaptive_ratio, restore_mask)

                fp32_param = to_float(merge_param_data) * adaptive_ratio + to_float(param.data) * (1 - adaptive_ratio)
                param.data[:] = fp32_param.to(param.dtype) 
        return model_to_update

    def _get_adaptive_q(self, logits):
        with torch.no_grad():
            entropy = softmax_entropy(logits)
            valid_idx = torch.where(entropy < self.entropy_margin)[0]
            if len(valid_idx) > 0:
                mean_entropy = entropy[valid_idx].mean()
            else:
                mean_entropy = torch.tensor(math.log(self.cfg.CORRUPTION.NUM_CLASS)).to(logits.device)
        
        base_q = self.q 
        max_entropy = math.log(self.cfg.CORRUPTION.NUM_CLASS)
        ratio = mean_entropy / (max_entropy + 1e-8)
        confidence = 1.0 - ratio
        scale_factor = (confidence ** 2) * 5.0 
        adaptive_q = base_q * scale_factor
        adaptive_q = torch.clamp(adaptive_q, min=1e-6, max=0.01)
        return adaptive_q

    @torch.no_grad()
    def bayesian_filtering(self, x, logits=None): 
        current_grams = self.gram_collector.collect_all_grams_once(x)

        if not self.historical_grams:
            self.historical_grams = current_grams
        else:
            for layer_idx, gram in current_grams.items():
                if layer_idx in self.historical_grams:
                    self.historical_grams[layer_idx] = (
                        (1 - self.gram_momentum) * self.historical_grams[layer_idx] + 
                        self.gram_momentum * gram
                    )
                else:
                    self.historical_grams[layer_idx] = gram
        
        all_grams = self.historical_grams
        current_q = self._get_adaptive_q(logits) if logits is not None else self.q
        
        recovered_model = deepcopy(self.hidden_model) 
        self._gram_weighted_merge(
            model_to_update=recovered_model,
            model_to_merge=self.src_model, 
            all_grams=all_grams,
            base_momentum=self.alpha,
            enable_restoration=True 
        )
        
        self.hidden_var = self.alpha ** 2 * self.hidden_var + current_q
        r = (1 - current_q)
        self.beta = r / (self.hidden_var + r)
        self.beta = max(self.beta, 0.89)
        self.beta = min(self.beta, 0.9999)
        self.hidden_var = self.beta * self.hidden_var
        
        self.hidden_model = recovered_model 
        self._gram_weighted_merge(
            model_to_update=self.hidden_model, 
            model_to_merge=self.model, 
            all_grams=all_grams,
            base_momentum=self.beta,
            enable_restoration=False
        )
        
        merge_source = recovered_model if self.post_type == "op" else self.hidden_model
        self._gram_weighted_merge(
            model_to_update=self.model, 
            model_to_merge=merge_source, 
            all_grams=all_grams,
            base_momentum=self.gamma,
            enable_restoration=False
        )

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.batch_index = -1
        # Use HeML configuration items
        self.sigma_f = self.cfg.ADAPTER.HeML.SIGMA_F
        self.reset_prototypes()
        self.hidden_var = 0
        self.ema_model.load_state_dict(self.model_state)
        self.hidden_model.load_state_dict(self.model_state)
        self.historical_grams = {} 
        log.info("HeML model reset.")

    def get_pcs_weights(self, strong_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prob_outputs = logits.softmax(1)
            cls1 = prob_outputs.argmax(dim=1)
            prob_outputs_strong = strong_logits.softmax(1)
            pcs = torch.gather(prob_outputs, 1, cls1.view(-1, 1)) - torch.gather(prob_outputs_strong, 1, cls1.view(-1, 1))
            pcs = pcs.view(-1)
        pcs = (pcs - pcs.min()) / (pcs.max() - pcs.min() + 1e-8)
        return pcs

def setup(model: nn.Module, cfg):
    log.info("Setup TTA method: HeML (Hybrid Empirical Model Learning) with Adaptive Bayesian Filtering Update")
    model = configure_model(model, cfg)
    tta_model = HeML(
        cfg,
        model
    )
    return tta_model