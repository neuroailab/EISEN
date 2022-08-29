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
                 excite=True,
                 inhibit=True,
                 project=False,
                 adj_thresh=0.5):
        super().__init__()
        self.num_iters = num_iters
        self.excite = excite
        self.inhibit = inhibit
        self.project = project
        self.adj_thresh = adj_thresh

        self.adj_e = None
        self.adj_i = None
        self.norm_factor_e = None
        self.norm_factor_i = None
        self.activation_converge = False

    def preprocess_inputs(self, h0, adj, activated):
        B, N, K = h0.shape

        # activated: binary tensor indicating nodes to start propagation from
        if activated is None: # randomly initialized at a point
            rand_idx = torch.randint(0, N, [B, ]).to(h0.device)
            activated = F.one_hot(rand_idx, num_classes=N).unsqueeze(-1).float()  # [BT, N, 1]
        else:
            activated = activated.reshape(B, N, 1)

        # create excitatory and inhibitory affinities, followed by thresholding
        if not isinstance(adj, SparseTensor): # dense tensor
            adj = adj.reshape(B, N, N)
            adj_e, adj_i = adj, 1.0 - adj
            if self.adj_thresh is not None:
                adj_e = self._threshold(adj_e, self.adj_thresh)
                adj_i = self._threshold(adj_i, self.adj_thresh)
            sample_mask = None
        else: # sparse tensor
            adj_e = adj
            adj_i = adj.copy().set_value_(1.0 - adj.storage.value())

            sample_mask = adj_e.copy()
            sample_mask = sample_mask.set_value_(torch.ones_like(sample_mask.storage.value()))

            if self.adj_thresh is not None:
                adj_e = self._threshold_sparse_tensor(adj_e, self.adj_thresh)
                adj_i = self._threshold_sparse_tensor(adj_i, self.adj_thresh)

        return h0, adj_e, adj_i, activated, sample_mask

    def forward(self, h0, adj, activated=None):
        """
        Function: Graph propagation to create the plateau map representation
        Input:
        - h0: initial hidden states (or equivalently, plateau map representation) with shape [B, N Q]
        - adj: affinity matrix with shape
        - activated: The graph propagation will start from the activated nodes
                   It can be None or binary tensor of shape [B, N].
                   If None, one node will be randomly selected as being activated
        Return:
        - plateau_map_list: a list of plateau maps of len self.num_iters. Each plateau map has shape [B, N, Q]
        """

        h0, adj_e, adj_i, activated, sample_mask = self.preprocess_inputs(h0, adj, activated)

        h = h0.clone()
        plateau_map_list = []
        running_activated = activated
        self.activation_converge = False

        # start graph propagation
        for it in range(self.num_iters):
            h, activated, running_activated = \
                self.propagate(h, adj_e, adj_i, activated, running_activated, sample_mask, it)
            plateau_map_list.append(h.reshape(h0.shape))

        return plateau_map_list

    def propagate(self, h, adj_e, adj_i, activated, running_activated, sample_mask, iter):
        B, N, D = h.shape

        # Graph propagation starts at a subset of activated nodes
        # If self.activation_converge is False, i.e. not all nodes are activated, \
        #   we need to apply masking to the affinities and compute the normalization factor accordingly.
        # We do so until all the nodes are activated, i.e.  self.activation_converge == True

        if not self.activation_converge:
            if isinstance(adj_e, SparseTensor):
                # apply the activation mask on the affinity tensors
                adj_e = adj_e.mul(activated.flatten()[None])
                adj_i = adj_i.mul(activated.flatten()[None])
                sample_mask = sample_mask.mul(activated.flatten()[None])

            # compute the normalization factors
            if not isinstance(adj_e, SparseTensor):
                norm_factor_e = torch.sum(adj_e.abs() * activated, dim=-2, keepdim=True).clamp(min=1.0).detach()
                norm_factor_i = torch.sum(adj_i.abs() * activated, dim=-2, keepdim=True).clamp(min=1.0).detach()
            else:
                norm_factor_e = adj_e.sum(1).reshape(B, N, 1).clamp(min=1.0).detach()
                norm_factor_i = adj_i.sum(1).reshape(B, N, 1).clamp(min=1.0).detach()

            self.norm_factor_e = norm_factor_e # [B,1,N]
            self.norm_factor_i = norm_factor_i # [B,1,N]
            self.adj_e = adj_e
            self.adj_i = adj_i

            self.activation_converge = activated.sum() == (B * N)
        else: # no update is required if all the nodes are activated
            adj_e = self.adj_e
            adj_i = self.adj_i
            norm_factor_e = self.norm_factor_e
            norm_factor_i = self.norm_factor_i
            sample_mask = None

        # [Excitation]
        if self.excite:
            if not isinstance(adj_e, SparseTensor):
                e_effects = torch.matmul(h.permute(0, 2, 1), adj_e * activated) / norm_factor_e
                e_effects = e_effects.permute(0, 2, 1)
            else:
                e_effects = adj_e.matmul(h.reshape(B * N, D))
                e_effects = e_effects.reshape(B, N, D) / norm_factor_e
            h = h + e_effects

        # [Inhibition]
        if self.inhibit:
            if not isinstance(adj_e, SparseTensor):
                i_effects = torch.matmul(h.permute(0, 2, 1), adj_i * activated) / norm_factor_i
                i_effects = i_effects.permute(0, 2, 1)
            else:
                i_effects = adj_i.matmul(h.reshape(B * N, D))
                i_effects = i_effects.reshape(B, N, D) / norm_factor_i

            proj = self._projection(h, i_effects) if self.project else i_effects
            h = h - proj

        h = self._relu_norm(h)

        # [Update activated nodes]
        if activated.sum() < B * N:
            if not isinstance(adj_e, SparseTensor):
                receivers = torch.max(torch.where(adj_e > adj_i, adj_e, adj_i) * activated, dim=1, keepdim=False)[0] > 0.5 # [B,N]
            else:
                assert sample_mask is not None
                receivers = sample_mask.max(dim=1) > 0.5
                receivers = receivers.reshape(B, N)

            running_activated = running_activated + receivers.unsqueeze(-1).float()
            activated = running_activated.clamp(max=1.0).detach()

        return h, activated, running_activated

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
    def _relu_norm(x, relu=True, norm=True, eps=1e-16):
        x = F.relu(x) if relu else x
        x = F.normalize(x + eps, p=2.0, dim=-1, eps=max([eps, 1e-12])) if norm else x
        return x

if __name__ == '__main__':
    pass
