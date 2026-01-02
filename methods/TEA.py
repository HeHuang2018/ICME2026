from copy import deepcopy
import os
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as log
from torchvision.utils import save_image
__all__ = ["setup"]

def init_random(bs, im_sz=32, n_ch=3):
    return torch.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


class EnergyModel(nn.Module):
    def __init__(self, model):
        super(EnergyModel, self).__init__()
        self.f = model

    def classify(self, x):
        penult_z = self.f(x)
        return penult_z
    
    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1), logits
        else:
            return torch.gather(logits, 1, y[:, None]), logits
        

def sample_p_0(reinit_freq, replay_buffer, bs, im_sz, n_ch, device, y=None):
    if len(replay_buffer) == 0:
        return init_random(bs, im_sz=im_sz, n_ch=n_ch), []
    buffer_size = len(replay_buffer)
    inds = torch.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds

    buffer_samples = replay_buffer[inds]
    random_samples = init_random(bs, im_sz=im_sz, n_ch=n_ch)
    choose_random = (torch.rand(bs) < reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(device), inds


def sample_q(f, replay_buffer, n_steps, sgld_lr, sgld_std, reinit_freq, batch_size, im_sz, n_ch, device, y=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()
    # get batch size
    bs = batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(reinit_freq=reinit_freq, replay_buffer=replay_buffer, bs=bs, im_sz=im_sz, n_ch=n_ch, device=device ,y=y)
    init_samples = deepcopy(init_sample)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    for k in range(n_steps):
        f_prime = torch.autograd.grad(f(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples, init_samples.detach()

class TEA(nn.Module):
    """TEA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, optimizer): 
        super().__init__()
        self.cfg = cfg
        self.energy_model = EnergyModel(model)
        self.replay_buffer = init_random(self.cfg.ADAPTER.TEA.BUFFER_SIZE, im_sz=self.cfg.ADAPTER.TEA.IMG_SIZE, n_ch=self.cfg.ADAPTER.TEA.NUM_CHANNEL)
        self.replay_buffer_old = deepcopy(self.replay_buffer)
        self.optimizer = optimizer
        self.steps = 1
        assert self.steps > 0, "TEA requires >= 1 step(s) to forward and update"
        self.episodic = False

        self.sgld_steps = self.cfg.ADAPTER.TEA.STEPS
        self.sgld_lr = self.cfg.ADAPTER.TEA.SGLD_LR
        self.sgld_std = self.cfg.ADAPTER.TEA.SGLD_STD
        self.reinit_freq = self.cfg.ADAPTER.TEA.REINIT_FREQ
        self.if_cond = self.cfg.ADAPTER.TEA.UNCOND
        
        self.n_classes = self.cfg.CORRUPTION.NUM_CLASS
        self.im_sz = self.cfg.ADAPTER.TEA.IMG_SIZE
        self.n_ch = self.cfg.ADAPTER.TEA.NUM_CHANNEL
        
        self.path = self.cfg.OUTPUT_DIR

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.energy_model, self.optimizer)

    def forward(self, x, y=None, adapt=True, counter=None, if_vis=False):
        if self.episodic:
            self.reset()
        
        if if_adapt:
            for i in range(self.steps):
                outputs = forward_and_adapt(x, self.energy_model, self.optimizer, 
                                            self.replay_buffer, self.sgld_steps, self.sgld_lr, self.sgld_std, self.reinit_freq,
                                            if_cond=self.if_cond, n_classes=self.n_classes)
                # if i % 1 == 0 and if_vis:
                #     visualize_images(path=self.path, replay_buffer_old=self.replay_buffer_old, replay_buffer=self.replay_buffer, energy_model=self.energy_model, 
                #                     sgld_steps=self.sgld_steps, sgld_lr=self.sgld_lr, sgld_std=self.sgld_std, reinit_freq=self.reinit_freq,
                #                     batch_size=100, n_classes=self.n_classes, im_sz=self.im_sz, n_ch=self.n_ch, device=x.device, counter=counter, step=i)
        else:
            self.energy_model.eval()
            with torch.no_grad():
                outputs = self.energy_model.classify(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.energy_model, self.optimizer,
                                 self.model_state, self.optimizer_state)

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

@torch.enable_grad()
def visualize_images(path, replay_buffer_old, replay_buffer, energy_model, 
                     sgld_steps, sgld_lr, sgld_std, reinit_freq,
                     batch_size, n_classes, im_sz, n_ch, device=None, counter=None, step=None):
    num_cols=10
    repeat_times = batch_size // n_classes
    y = torch.arange(n_classes).repeat(repeat_times).to(device) 
    x_fake, _ = sample_q(energy_model, replay_buffer, n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=y)
    images = x_fake.detach().cpu()
    save_image(images , os.path.join(path, 'sample.png'), padding=2, nrow=num_cols)

    num_cols=40
    images_init = replay_buffer_old.cpu()
    images = replay_buffer.cpu() 
    images_diff = replay_buffer.cpu() - replay_buffer_old.cpu()
    if step == 0:
        save_image(images_init , os.path.join(path, 'buffer_init.png'), padding=2, nrow=num_cols)
    save_image(images , os.path.join(path, 'buffer-'+str(counter)+"-"+str(step)+'.png'), padding=2, nrow=num_cols) # 
    save_image(images_diff , os.path.join(path, 'buffer_diff.png'), padding=2, nrow=num_cols)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, energy_model, optimizer, replay_buffer, sgld_steps, sgld_lr, sgld_std, reinit_freq, if_cond=False, n_classes=10):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    batch_size=x.shape[0]
    n_ch = x.shape[1]
    im_sz = x.shape[2]
    device = x.device
    
    if if_cond == 'uncond':
        x_fake, _ = sample_q(energy_model, replay_buffer, 
                             n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, 
                             batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=None)
    elif if_cond == 'cond':
        y = torch.randint(0, n_classes, (batch_size,)).to(device)
        x_fake, _ = sample_q(energy_model, replay_buffer, 
                             n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, 
                             batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=y)

    # forward
    out_real = energy_model(x)
    energy_real = out_real[0].mean()
    energy_fake = energy_model(x_fake)[0].mean()

    # adapt
    loss = (- (energy_real - energy_fake)) 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    outputs = energy_model.classify(x)

    return outputs

def collect_params(model: nn.Module):
    """Collect parameters from normalization layers that require gradients.
    """
    params = []
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    params.append(param)
                    names.append(f"{name}.{param_name}")
    return params, names

def configure_model(model):
    model.train()
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, nn.BatchNorm1d):
            m.train()
            m.requires_grad_(True)
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model

def setup(model, cfg):
    log.info("Setup TTA method: TEA")

    model = configure_model(model)
    params, param_names = collect_params(model)
    if cfg.OPTIM.METHOD == "SGD":
        optimizer = torch.optim.SGD(
            params, 
            lr=float(cfg.OPTIM.LR),
            dampening=cfg.OPTIM.DAMPENING,
            momentum=float(cfg.OPTIM.MOMENTUM),
            weight_decay=float(cfg.OPTIM.WD),
            nesterov=cfg.OPTIM.NESTEROV
        )
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = torch.optim.Adam(
            params, 
            lr=float(cfg.OPTIM.LR),
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=float(cfg.OPTIM.WD)
        )
    TTA_model = TEA(
        cfg,
        model, 
        optimizer
    )
    return TTA_model