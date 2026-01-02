import re
import os
import os.path as osp
import warnings
import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from loguru import logger as log
from torchvision import transforms
from core.model.build import ResNetDomainNet126
from torch.nn.utils.weight_norm import WeightNorm
import random
import errno
import pandas as pd
import numpy as np
from loguru import logger as log
import shutil

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

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

class MetricsCollector:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, cfg):
        self.model = model # Keep a reference if needed for collecting current params
        self.optimizer = optimizer
        self.cfg = cfg

        # Store initial adaptable weights
        initial_adaptable_params, _ = collect_params(self.model)
        if not initial_adaptable_params:
            log.warning(f"MetricsCollector: Weight change metrics might be zero or incorrect.")
            self.initial_adaptable_weights_tensors = []
        else:
            self.initial_adaptable_weights_tensors = [p.clone().detach() for p in initial_adaptable_params]
        
        self.batch_index = 0
        self.metrics_to_save = {}
        self.bn_activation_outputs_for_metric = [] # Populated by hooks
        self.hook_handles = []
        
        self._register_bn_activation_hooks()

    def _register_bn_activation_hooks(self):
        self._remove_bn_activation_hooks() # Clear any existing hooks first
        for name, module in self.model.named_modules():
            # Currently hardcoded for BN, could be made more generic if needed
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                handle = module.register_forward_hook(
                    lambda m, i, o, layer_name=name: self._bn_activation_hook_fn(m, i, o, layer_name)
                )
                self.hook_handles.append(handle)
        log.info(f"MetricsCollector: Registered {len(self.hook_handles)} BN activation hooks.")

    def _remove_bn_activation_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def _bn_activation_hook_fn(self, module, input_data, output_data, layer_name):
        """Hook to capture the L0 norm of activations before BN layers."""
        with torch.no_grad():
            if isinstance(input_data, tuple): # Input can be a tuple
                input_tensor = input_data[0]
            else:
                input_tensor = input_data

            if input_tensor.ndim == 4:
                activation_stats = input_tensor.abs().mean(dim=[0, 2, 3]) 
            elif input_tensor.ndim == 3:
                activation_stats = input_tensor.abs().mean(dim=[0,1])
            elif input_tensor.ndim == 2: # e.g., (N, C) for Linear -> BN1D
                activation_stats = input_tensor.abs().mean(dim=0) # Avg over batch
            else:
                log.warning(f"BN Activation Hook: Unsupported input tensor ndim {input_tensor.ndim} for layer {layer_name}")
                return
            self.bn_activation_outputs_for_metric.append({
                'name': layer_name,
                'activation_abs_mean': activation_stats.cpu().numpy() # Store per-channel means
            })

    def clear_batch_hook_data(self):
        self.bn_activation_outputs_for_metric.clear()

    def collect_and_save_metrics(self, current_loss: float, y: torch.Tensor, outputs: torch.Tensor, entropy: torch.Tensor):
        metrics = {}

        # 1. BN Weight Change
        current_bn_params, _ = collect_params(self.model)
        sum_sq_diff_norm = 0.0
        sum_sq_initial_norm = 0.0
        
        if not self.initial_adaptable_weights_tensors and not current_bn_params:
            # log.debug("No BN parameters to calculate weight change.") # Can be noisy
            metrics['bn_weight_change_normalized_l2'] = 0.0
        elif len(self.initial_adaptable_weights_tensors) != len(current_bn_params):
            log.error(f"MetricsCollector: Mismatch in initial ({len(self.initial_adaptable_weights_tensors)}) "
                      f"and current ({len(current_bn_params)}) adaptable parameters for weight change calculation.")
            metrics['bn_weight_change_normalized_l2'] = -1.0 
        else:
            for p_initial, p_current in zip(self.initial_adaptable_weights_tensors, current_bn_params):
                diff = p_initial.to(p_current.device) - p_current 
                sum_sq_diff_norm += torch.norm(diff, p=2).item()**2
                sum_sq_initial_norm += torch.norm(p_initial.to(p_current.device), p=2).item()**2
            metrics['bn_weight_change_normalized_l2'] = \
                (np.sqrt(sum_sq_diff_norm) / (np.sqrt(sum_sq_initial_norm) + 1e-9)) if sum_sq_initial_norm > 1e-9 else 0.0
        
        # 2. BN Gradient Norms (from optimizer's parameters)
        current_bn_grads_l0_list = []
        current_bn_grads_l1_list = []
        current_bn_grads_min_max_list = []

        # Iterate over params in optimizer, assuming they are the ones being adapted
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_data = p.grad.detach()
                    current_bn_grads_l0_list.append(torch.sum(grad_data != 0).item())
                    current_bn_grads_l1_list.append(torch.norm(grad_data, p=1).item())
                    # Normalize gradients per parameter tensor
                    g_min = grad_data.min()
                    g_max = grad_data.max()
                    if (g_max - g_min).abs() > 1e-8 : # Avoid division by zero if grad is flat
                        bn_grads_min_max = (grad_data - g_min) / (g_max - g_min)
                        current_bn_grads_min_max_list.extend(bn_grads_min_max.flatten().cpu().numpy())
                    elif grad_data.numel() > 0 : # if not flat but all same value (e.g. all zeros)
                         current_bn_grads_min_max_list.extend(torch.zeros_like(grad_data.flatten()).cpu().numpy())


        # metrics['bn_gradient_normalized_l0_norm'] = np.mean(current_bn_grads_l0_list) if current_bn_grads_l0_list else 0.0
        metrics['bn_gradient_normalized_l1_norm'] = np.mean(current_bn_grads_l1_list) if current_bn_grads_l1_list else 0.0
        # metrics['bn_gradient_normalized_min_max'] = np.mean(current_bn_grads_min_max_list) if current_bn_grads_min_max_list else 0.0 # Or handle case of empty list differently

        # 3. Other metrics
        metrics['loss'] = current_loss 
        metrics['outputs'] = outputs.detach().cpu().numpy()
        metrics['entropy'] = entropy.detach().cpu().numpy()
        metrics['y'] = y.cpu().numpy()

        # 4. BN Activation Stats
        if self.bn_activation_outputs_for_metric:
            # Average of mean absolute activations across all BN layers
            all_layer_avg_activations = [np.mean(data['activation_abs_mean']) for data in self.bn_activation_outputs_for_metric]

            metrics['bn_input_activation_abs_mean_overall'] = np.mean(all_layer_avg_activations) if all_layer_avg_activations else 0.0

        else:
            metrics['bn_input_activation_abs_mean_overall'] = 0.0
        
        self.clear_batch_hook_data() # Clear after saving
        self.metrics_to_save[self.batch_index] = metrics

        save_interval = self.cfg.SAVE_INTERVAL if hasattr(self.cfg, 'SAVE_INTERVAL') else (79 * 15) 
        if self.batch_index > 0 and self.batch_index % save_interval == 0:
            try:
                path_template = "/home/xionghaoyu/code/xhy/CTTA/outputs/{dataset}_{method_name}_metrics_{sampler_type}_{gamma}_{note}_{batch_idx}.npy"
                
                filename = path_template.format(
                    dataset=self.cfg.CORRUPTION.DATASET,
                    method_name=self.cfg.ADAPTER.NAME,
                    sampler_type=self.cfg.LOADER.SAMPLER.TYPE,
                    gamma=self.cfg.LOADER.SAMPLER.GAMMA,
                    note=self.cfg.NOTE,
                    batch_idx=self.batch_index 
                )
                np.save(filename, self.metrics_to_save)
                log.info(f"MetricsCollector: Saved metrics up to batch {self.batch_index} to {filename}")
            except Exception as e:
                log.error(f"MetricsCollector: Failed to save metrics at batch {self.batch_index}. Error: {e}")
        
        self.batch_index += 1

    def close(self):
        self._remove_bn_activation_hooks()
        log.info("MetricsCollector closed, BN activation hooks removed.")


def save_and_rename_file(source_path, destination_folder, new_name):
    os.makedirs(destination_folder, exist_ok=True)
    
    new_path = os.path.join(destination_folder, new_name)
    
    shutil.copy(source_path, new_path)
    
    print(f"File saved and renamed to: {new_path}")

def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 print_every: int):

    num_correct = 0.
    num_samples = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            output = model([img.cuda() for img in imgs]) if isinstance(imgs, list) else model(imgs.cuda())
            predictions = output.argmax(1)
            num_correct += (predictions == labels.cuda()).float().sum()
            # track progress
            num_samples += imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
            # if print_every > 0 and (i+1) % print_every == 0:
            #     log.info(f"#batches={i+1:<6} #samples={num_samples:<9} error = {1 - num_correct / num_samples:.2%}")

            if dataset_name == "ccc" and num_samples >= 7500000:
                break

    accuracy = num_correct.item() / num_samples
    return accuracy

def save_df(new_results, path):
    try:
        all_results_df = pd.read_csv(path)
        all_results_df = all_results_df.append(new_results, ignore_index=True)
    except:
        mkdir(osp.dirname(path))
        all_results_df = pd.DataFrame(new_results, index=[0])
    all_results_df.to_csv(path, index=False)
    return all_results_df

def seed_everything(seed):
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            log.info(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                log.info(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)
    if not (min_seed_value <= seed <= max_seed_value):
        log.info(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def _select_seed_randomly(min_seed_value, max_seed_value):
    return random.randint(min_seed_value, max_seed_value)  # noqa: S311

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
def check_isfile(fpath):
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile

def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)
    return module

def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])
        else:
            setattr(module, names[i], value)
