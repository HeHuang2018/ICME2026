import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
from loguru import logger as log

def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat

def pairwise_cosine_sim_np(a, b):
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    a_norm[a_norm == 0] = 1
    b_norm[b_norm == 0] = 1

    a_normalized = a / a_norm
    b_normalized = b / b_norm

    mat = np.dot(a_normalized, b_normalized.T)
    return 1 - mat

class Prototype:
    def __init__(self, C=1000, dim=512,batch_size=64):
        self.feats_center = None
        self.logits_center = None
        self.C = C
        self.feats_pro = torch.zeros(C, dim).cuda()
        self.batch_index = -1
        self.class_weight = torch.ones(C).cuda()

    @torch.no_grad()
    def update(self, logits, feats, momentum = 0.9,norm=True):
        lbls = torch.argmax(logits, dim=1)
        if self.feats_center is None:
            self.feats_center = feats.mean(dim=0, keepdim=True)
            self.logits_center = logits.softmax(1).mean(0, keepdim=True)
        else:
            self.feats_center = self.feats_center * momentum + feats.mean(dim=0, keepdim=True) * (1 - momentum) 
            self.logits_center = self.logits_center * momentum + logits.softmax(1).mean(0, keepdim=True) * (1 - momentum)

        for i_cls in torch.unique(lbls):
            feats_i = feats[lbls == i_cls, :]
            feats_i_center = feats_i.mean(dim=0, keepdim=True)
            self.feats_pro[i_cls, :] = self.feats_pro[i_cls, :] * momentum + feats_i_center * (1 - momentum)

        # self.feats_pro = self.feats_pro.mean(0, keepdim=True).repeat(self.C, 1)
        if norm:
            self.feats_pro = F.normalize(self.feats_pro)
            self.feats_center = F.normalize(self.feats_center)

        self.momentum = momentum
        self.batch_index += 1

    @torch.enable_grad()
    def center_loss_cls(self, centers, feats, Q_st, mask, weights, note):
        ot_weights, ot_label = Q_st.t().max(dim=1)
        
        feats = feats[~mask]
        weights = weights[~mask]
        ot_weights = ot_weights[~mask]
        
        # batch_size = feats.size(0)
        centers = F.normalize(centers)
        feats = F.normalize(feats)
        dist = 1 - feats @ centers.t()
        dist = dist.clamp(min=1e-12, max=1e+12)
        dist = dist.sum(-1)
        try:
            ot_weights = (ot_weights - ot_weights.min()) / (ot_weights.max() - ot_weights.min() + 1e-12)
            # exp_weights = torch.exp(ot_weights * weights) ** 1.0
            exp_weights = torch.exp(ot_weights * weights * 0.001)
            dist = dist * exp_weights
        except Exception as e:
            # log.info(f"==>> {weights} {ot_weights}")
            pass
        dist = dist
        # dist = dist[~mask]

        loss = dist.sum() / len(dist)
        return loss

    def ot_mapping(self, sim):
        ns, nt = sim.shape
        # if self.batch_index == 0:
        self.alpha, self.beta = np.ones((ns,)) / ns, np.ones((nt,)) / nt

        if len(self.beta) != nt:
            self.beta = np.ones((nt,)) / nt

        Q_st = ot.unbalanced.sinkhorn_stabilized_unbalanced(self.alpha, self.beta, sim, reg=1, reg_m=1)
        Q_st = torch.from_numpy(Q_st).float().cuda()

        sum_pi = torch.sum(Q_st)
        Q_st_bar = Q_st/sum_pi
        new_beta = torch.sum(Q_st_bar,0).cpu().numpy()


        Q_anchor = Q_st_bar
        wt_i, pseudo_label = torch.max(Q_anchor.t(), 1)
        ws_j = torch.sum(Q_st_bar)
        conf_label_idx = torch.where(wt_i > 1/Q_st_bar.size(1))


        # # update class weight and target weight by plan pi
        # batch_size = Q_st_bar.size(0)
        # plan = Q_st * batch_size

        # self.alpha = self.momentum * self.alpha + (1-self.momentum) * np.array([ws_j.cpu().numpy()])
        # self.beta = self.momentum * self.beta + (1-self.momentum) * new_beta
        return Q_st
        
    def OT(self, feats_pro, feat_tu_w):
        with torch.no_grad():
            sim = 1 - pairwise_cosine_sim(feats_pro, feat_tu_w)

        Q_st = self.ot_mapping(
            sim.data.cpu().numpy().astype(np.float64)
        )
        return Q_st
