import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import os
import time
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt


class GraphPropagation(nn.Module):
    def __init__(self,
                 num_iters=15,
                 project=False,
                 adj_thresh=None,
                 stop_gradient=False,
                 excite=True,
                 inhibit=True,
                 propagate_activation=True,
                 stop_gradient_h=False,):
        super().__init__()
        self.adj_thresh = adj_thresh
        self.stop_gradient = stop_gradient
        self.num_iters = num_iters
        self.excite = excite
        self.inhibit = inhibit
        self.project = project
        self.stop_gradient_h = stop_gradient_h
        self.propagate_activation = propagate_activation

    def preprocess_inputs(self, h0, adj, activated):
        B, N, K = h0.shape

        # reshape activated
        if activated is None:
            rand_idx = torch.randint(0, N, [B, ]).to(h0.device)
            activated = F.one_hot(rand_idx, num_classes=N).unsqueeze(-1).float()  # [B, N, 1]
        else:
            activated = activated.reshape(B, N, 1)

        # process affinity (shape: [BT, N, N])
        if isinstance(adj, tuple):
            assert isinstance(adj[0], SparseTensor) and isinstance(adj[1], SparseTensor)
            adj_e, adj_i = adj
        else:
            assert isinstance(adj, SparseTensor)
            adj_e = adj
            adj_i = adj.copy()
            adj_i = adj_i.set_value_(1.0 - adj.storage.value())

        if self.adj_thresh is not None:
            adj_e = adj_e.set_value_(self._threshold(adj_e.storage.value(), self.adj_thresh))
            adj_i = adj_i.set_value_(self._threshold(adj_i.storage.value(), self.adj_thresh))

            sample_mask = adj_e.copy()
            sample_mask = sample_mask.set_value_(torch.ones_like(sample_mask.storage.value()))

        self.adj_e, self.adj_i = adj_e, adj_i

        return h0, adj_e, adj_i, activated, sample_mask

    def forward(self, h0, adj, activated=None):

        preds, adj_e, adj_i, activated, sample_mask = self.preprocess_inputs(h0, adj, activated)
        running_activated = activated
        self.activated_converge = False

        # start propagation
        preds_list = []
        h = preds.clone()

        for it in range(self.num_iters):
            out = self.propagate(h, adj_e, adj_i, activated, running_activated, sample_mask, it)
            h, activated, running_activated, intermediates = out
            preds_list.append(h)

        return preds_list

    def propagate(self, h, adj_e, adj_i, activated, running_activated, sample_mask, iter):
        B, N, D = h.shape

        if self.stop_gradient_h:
            h = h.detach()

        intermediates = {}
        if not self.activated_converge:

            save_adj_e = adj_e.copy()
            save_adj_i = adj_i.copy()

            if self.propagate_activation:
                adj_e = adj_e.mul(activated.flatten()[None])
                adj_i = adj_i.mul(activated.flatten()[None])
                sample_mask = sample_mask.mul(activated.flatten()[None])

                save_adj_e = save_adj_e#.set_value_(save_adj_e.storage.value().abs())
                save_adj_i = save_adj_i#.set_value_(save_adj_i.storage.value().abs())

                save_adj_e = save_adj_e.mul(activated.flatten()[None])
                save_adj_i = save_adj_i.mul(activated.flatten()[None])
                n_senders_e = save_adj_e.sum(1).reshape(B, N, 1).clamp(min=1.0).detach()
                n_senders_i = save_adj_i.sum(1).reshape(B, N, 1).clamp(min=1.0).detach()

            self.n_senders_e = n_senders_e
            self.n_senders_i = n_senders_i
            self.adj_e = adj_e
            self.adj_i = adj_i

            if activated.sum() == B * N:
                self.activated_converge = True

        else:
            n_senders_e = self.n_senders_e
            n_senders_i = self.n_senders_i
            adj_e = self.adj_e
            adj_i = self.adj_i
            sample_mask = None

        if self.excite:

            e_effects = adj_e.matmul(h.reshape(B * N, D))
            e_effects = e_effects.reshape(B, N, D) / n_senders_e
            h = h + e_effects

        if self.inhibit:

            i_effects = adj_i.matmul(h.reshape(B * N, D))
            i_effects = i_effects.reshape(B, N, D) / n_senders_i

            proj = self._projection(h, i_effects) if self.project else i_effects
            h = h - proj

        h = self._relu_norm(h)

        # update activated
        if self.propagate_activation and activated.sum() < B * N:
            assert sample_mask is not None
            receivers = sample_mask.max(dim=1) > 0.5
            receivers = receivers.reshape(B, N)

            running_activated = running_activated + receivers.unsqueeze(-1).float()
            activated = running_activated.clamp(max=1.0).detach()

        return h, activated, running_activated, intermediates


    @staticmethod
    def _threshold(x, thresh):
        return x * (x > thresh).float()

    @staticmethod
    def _threshold_sparse_tensor(x, thresh):
        row, col, value = x.coo()
        valid = value > thresh
        sparse_size = [x.size(0), x.size(1)]
        output = SparseTensor(row=row[valid],col=col[valid],value=value[valid],sparse_sizes=sparse_size)
        return output

    @staticmethod
    def _projection(v, u, eps=1e-12):
        u_norm = torch.sum(u * u, -1, keepdims=True)
        dot_prod = torch.sum(v * u, -1, keepdims=True)
        proj = (dot_prod / (u_norm + eps)) * u
        return proj

    @staticmethod
    def _relu_norm(x, relu=True, norm=True, eps=1e-8):
        x = F.relu(x) if relu else x
        x = F.normalize(x + eps, p=2.0, dim=-1, eps=max([eps, 1e-12])) if norm else x
        return x

if __name__ == '__main__':
    pass
