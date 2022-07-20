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
                 random_init=False,
                 beta=1.0,
                 excite=True,
                 inhibit=True,
                 update_activated=True,
                 push=False,
                 damp=False,
                 sharpen=False,
                 affinity_var=False,
                 use_rbp=False,
                 num_rbp_iters=20,
                 size=None,
                 valid_thresh=None,
                 eps=1e-8,
                 subsample_affinity=False,
                 norm_type='adjacent',
                 propagate_activation=True,
                 stop_gradient_h=False,
                 edge_update=False):
        super().__init__()
        self.adj_thresh = adj_thresh
        self.stop_gradient = stop_gradient
        self.random_init = random_init
        self.num_iters = num_iters
        self.excite = excite
        self.inhibit = inhibit
        self.project = project
        self.update_activated = update_activated
        self.beta = beta
        self.push = push
        self.damp = damp
        self.sharpen = sharpen
        self.use_rbp = use_rbp
        self.num_rbp_iters = num_rbp_iters
        self.valid_thresh = valid_thresh
        self.activated_adj_e = None
        self.activated_adj_i = None
        self.n_senders_e = None
        self.n_senders_i = None
        self.scatter_adj_e = None
        self.scatter_adj_i = None
        self.eps = eps
        self.stop_gradient_h = stop_gradient_h
        if affinity_var:
            print('Initialize affinity variable...')
            assert size is not None
            N = size[0] * size[1]
            self.key = torch.nn.Parameter(torch.cuda.FloatTensor(1, N, 32).normal_(), requires_grad=True)
            self.query = torch.nn.Parameter(torch.cuda.FloatTensor(1, N, 32).normal_(), requires_grad=True)
            self.affinity_var = True
        else:
            self.affinity_var = False
        self.propagate_activation = propagate_activation
        self.norm_type = norm_type

        # edge_update = self.edge_update
        # if self.edge_update:
        #     self.slope_conv = nn.Conv2d(1024, 1, bias=True, padding='same')
        #     self.adj_proj_conv = nn.Conv2d(1024, 1024, bias=True, padding='same')
        #     self.thres_conv = nn.Conv2d(1024, 1, bias=True, padding='same')

    def preprocess_inputs(self, h0, adj, activated):
        # reshape h0 from [B, T, H, W, K] to [BT, HW, K]
        B, N, K = h0.shape

        # reshape activated
        if activated is None:
            rand_idx = torch.randint(0, N, [B, ]).to(h0.device)
            activated = F.one_hot(rand_idx, num_classes=N).unsqueeze(-1).float()  # [BT, N, 1]

        else:
            activated = activated.reshape(B, N, 1)

        # process affinity (shape: [BT, N, N])
        if not isinstance(adj, SparseTensor):
            if self.affinity_var:    # setting affinity as a trainable variable
                adj = torch.matmul(self.key, self.query.permute(0, 2, 1)) / (32 ** 0.5)
                adj = (adj + adj.permute(0, 2, 1)) * 0.5
                adj = adj.sigmoid()
            else:
                adj = adj.reshape(B, N, N)

            # split affinities into excitatory and inhibitory
            adj_e, adj_i = adj, 1.0 - adj
            if self.adj_thresh is not None:
                adj_e = self._threshold(adj_e, self.adj_thresh)
                adj_i = self._threshold(adj_i, self.adj_thresh)

            if self.valid_thresh is not None:
                valid = adj > self.valid_thresh
                self.adj_valid = valid
                adj_e, adj_i = adj_e * valid, adj_i * valid
            self.adj_e, self.adj_i = adj_e, adj_i
        else:
            adj_e = adj
            adj_i = adj.copy()
            adj_i = adj_i.set_value_(1.0 - adj.storage.value())

            if self.adj_thresh is not None:
                # save_adj_e = adj_e.copy()
                # save_adj_i = adj_i.copy()


                sample_mask = adj_e.copy()
                sample_mask = sample_mask.set_value_(torch.ones_like(sample_mask.storage.value()))

                # adj_e = adj_e.set_value_(self._threshold(adj_e.storage.value(), self.adj_thresh))
                # adj_i = adj_i.set_value_(self._threshold(adj_i.storage.value(), self.adj_thresh))

                adj_e = self._threshold_sparse_tensor(adj_e, self.adj_thresh)
                adj_i = self._threshold_sparse_tensor(adj_i, self.adj_thresh)

                # assert torch.equal(adj_e.set_value_(self._threshold(adj_e.storage.value(), self.adj_thresh)).to_dense(), self._threshold_sparse_tensor(adj_e, self.adj_thresh).to_dense()), pdb.set_trace()
                # assert torch.equal(adj_i.set_value_(self._threshold(adj_i.storage.value(), self.adj_thresh)).to_dense(),
                #                    self._threshold_sparse_tensor(adj_i, self.adj_thresh).to_dense()), pdb.set_trace()

                # def vis(x, y):
                #     idx = x * 128 + y
                #     plt.subplot(1, 4, 1)
                #     plt.imshow(save_adj_e.to_dense()[idx].reshape(128, 128).cpu())
                #     plt.subplot(1, 4, 2)
                #     plt.imshow(adj_e.to_dense()[idx].reshape(128, 128).cpu())
                #     plt.subplot(1, 4, 3)
                #     plt.imshow(save_adj_i.to_dense()[idx].reshape(128, 128).cpu())
                #     plt.subplot(1, 4, 4)
                #     plt.imshow(adj_i.to_dense()[idx].reshape(128, 128).cpu())
                #     plt.show()
                #     plt.close()
                # pdb.set_trace()
            self.adj_e, self.adj_i = adj_e, adj_i

            # Note: activated masking and transpose has been applied when creating sparse tensor
            # adj_e = adj_e.mul(activated.reshape(B * N, 1))
            # adj_i = adj_i.mul(activated.reshape(B * N, 1))
            # adj_e, adj_i = adj_e.t(), adj_i.t()


        return h0, adj_e, adj_i, activated, sample_mask

    def forward(self, h0, adj, activated=None):
        B, N, K = h0.shape
        preds, adj_e, adj_i, activated, sample_mask = self.preprocess_inputs(h0, adj, activated)
        running_activated = activated
        self.activated_converge = False
        # start propagation
        preds_list = []
        h = preds.clone()
        intermediates_list = []
        if self.use_rbp:
            with torch.no_grad():
                for it in range(self.num_iters - 1):
                    h, activated = self.propagate_with_full_affinity(
                        h, adj_e, adj_i, activated, activated, it)
            h_2nd_last = h.detach().requires_grad_()
            it += 1
            h_last, activated = self.propagate_with_full_affinity(
                h_2nd_last, adj_e, adj_i, activated, activated, it)
            h = DummyKP.apply(h_2nd_last, h_last, self.num_rbp_iters)
            preds_list.append(h)

        else:  # BPTT
            for it in range(self.num_iters):
                out = self.propagate_with_full_affinity(h, adj_e, adj_i, activated, running_activated, sample_mask, it)
                h, activated, running_activated, intermediates = out
                preds_list.append(h)
                intermediates_list.append(intermediates)

                # # compute updates,
                # h, adj_e, adj_i = self.update(h, adj_e, adj_i, adj_local)
            h_2nd_last = h_last = None

        if self.sharpen:
            preds_list = [(h * self.beta).softmax(-1) for h in preds_list]

        # postprocess
        preds_list = [preds.reshape(B, N, K) for preds in preds_list]

        return preds_list, None, None

    def propagate_with_full_affinity(self, h, adj_e, adj_i, activated, running_activated, sample_mask, iter):

        B, N, D = h.shape
        # how many sender neurons
        # n_senders_e = torch.sum(adj_e.abs() * activated, dim=-2, keepdim=True).clamp(min=1.0).detach()  # [B,1,N]
        # n_senders_i = torch.sum(adj_i.abs() * activated, dim=-2, keepdim=True).clamp(min=1.0).detach()  # [B,1,N]

        # print('change normalization')
        #
        # n_senders_e = n_senders_i = 100.
        # normalization constant
        # assert adj_e.min() >= 0 and adj_e.max() <= 1, pdb.set_trace()
        # assert torch.unique(activated).shape[0] <= 2, ("expect activated to be binary", torch.unique(activated))

        if self.stop_gradient_h:
            h = h.detach()

        intermediates = {}
        if not self.activated_converge:

            # if isinstance(adj_e, SparseTensor):
            #     save_adj_e = adj_e.copy()
            #     save_adj_i = adj_i.copy()

            if self.propagate_activation and isinstance(adj_e, SparseTensor):
                adj_e = adj_e.mul(activated.flatten()[None])
                adj_i = adj_i.mul(activated.flatten()[None])
                sample_mask = sample_mask.mul(activated.flatten()[None])

            if not isinstance(adj_e, SparseTensor):
                if self.norm_type == 'activated':
                    n_senders_e = n_senders_i = activated.sum(dim=-2, keepdim=True).detach()
                elif self.norm_type == 'adjacent':
                    n_senders_e = torch.sum(adj_e.abs() * activated, dim=-2, keepdim=True).clamp(min=1.0).detach()  # [B,1,N]
                    n_senders_i = torch.sum(adj_i.abs() * activated, dim=-2, keepdim=True).clamp(min=1.0).detach()  # [B,1,N]
                elif self.norm_type == 'adjacent_clamped':
                    n_senders_e = torch.sum(adj_e.clamp(min=0., max=1.) * activated, dim=-2, keepdim=True).clamp(min=1.0).detach()  # [B,1,N]
                    n_senders_i = torch.sum(adj_i.clamp(min=0., max=1.) * activated, dim=-2, keepdim=True).clamp(min=1.0).detach()  # [B,1,N]
                elif self.norm_type == 'activated_local':
                    n_senders_e = n_senders_i = torch.sum(self.adj_valid * activated, dim=-2, keepdim=True).clamp(min=1.0).detach() # [B,1,N]
                else:
                    raise ValueError
            else:
                # Sparse implementation
                if self.norm_type == 'activated':
                    n_senders_e = n_senders_i = activated.sum(dim=-2, keepdim=True).detach()
                elif self.norm_type in ['adjacent', 'adjacent_clamped']:
                    if self.norm_type == 'adjacent':
                        pass
                        # save_adj_e = save_adj_e#.set_value_(save_adj_e.storage.value().abs())
                        # save_adj_i = save_adj_i#.set_value_(save_adj_i.storage.value().abs())
                    elif self.norm_type == 'adjacent_clamped':
                        save_adj_e = save_adj_e.set_value_(save_adj_e.storage.value().clamp(min=0., max=1.))
                        save_adj_i = save_adj_i.set_value_(save_adj_i.storage.value().clamp(min=0., max=1.))

                    save_adj_e = adj_e # save_adj_e.mul(activated.flatten()[None])
                    save_adj_i = adj_i # save_adj_i.mul(activated.flatten()[None])
                    n_senders_e = save_adj_e.sum(1).reshape(B, N, 1).clamp(min=1.0).detach()
                    n_senders_i = save_adj_i.sum(1).reshape(B, N, 1).clamp(min=1.0).detach()
                elif self.norm_type == 'activated_local':
                    local_mask = save_adj_e.set_value_(torch.ones_like(save_adj_e.storage.value()))
                    local_activated = local_mask.mul(activated.flatten()[None])
                    n_senders_e = n_senders_i = local_activated.sum(1).reshape(B, N, 1).clamp(min=1.0).detach()
                else:
                    raise ValueError


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
            if not isinstance(adj_e, SparseTensor):
                e_effects = torch.matmul(h.permute(0, 2, 1), adj_e * activated) / n_senders_e
                e_effects = e_effects.permute(0, 2, 1)
            else:
                e_effects = adj_e.matmul(h.reshape(B * N, D))
                e_effects = e_effects.reshape(B, N, D) / n_senders_e
            h = h + e_effects


        if self.inhibit:
            if not isinstance(adj_e, SparseTensor):
                i_effects = torch.matmul(h.permute(0, 2, 1), adj_i * activated) / n_senders_i
                i_effects = i_effects.permute(0, 2, 1)
            else:
                i_effects = adj_i.matmul(h.reshape(B * N, D))
                i_effects = i_effects.reshape(B, N, D) / n_senders_i

            proj = self._projection(h, i_effects) if self.project else i_effects
            h = h - proj

        h = self._relu_norm(h, eps=self.eps)
        # update activated

        if self.propagate_activation and activated.sum() < B * N:
            if not isinstance(adj_e, SparseTensor):
                receivers = torch.max(torch.where(adj_e > adj_i, adj_e, adj_i) * activated, dim=1, keepdim=False)[0] > 0.5 # [B,N]
            else:
                # Warning: tensor has already been transposed and activated has been multiplied
                # row, col, val_e = adj_e.coo()
                # _, _, val_i = adj_i.coo()
                # max_val = torch.maximum(val_e, val_i)
                # max_adj = SparseTensor(row=row, col=col, value=max_val, sparse_sizes=[B*N, B*N])
                # receivers = max_adj.max(dim=1) > 0.5
                # receivers = receivers.reshape(B, N)
                assert sample_mask is not None
                receivers = sample_mask.max(dim=1) > 0.5
                receivers = receivers.reshape(B, N)

            running_activated = running_activated + receivers.unsqueeze(-1).float()
            activated = running_activated.clamp(max=1.0).detach()

        # if self.update_activated:
        #     raise ValueError('Use propagated activated instead')
        #     receivers = torch.max(torch.where(adj_e > adj_i, adj_e, adj_i) * activated, dim=1, keepdim=False)[0] > 0.5 # [B,N]
        #     running_activated = running_activated + receivers.unsqueeze(-1).float()
        #     activated = running_activated.clamp(max=1.0).detach()

        # avg = torch.sum((h * self.beta).softmax(-1) * activated, dim=1, keepdim=True) / (torch.sum(activated, dim=1, keepdim=True) + 1e-6) # [B,1,Q]

        if self.push:
            free = ((1. - avg) * self.beta).softmax(-1)
            p_effects = n_senders_i * activated.permute(0, 2, 1)

            p_effects = p_effects * ((1. - activated) * free).permute(0, 2, 1)
            h += p_effects.permute(0, 2, 1)
            h = self._relu_norm(h, eps=self.eps)

        if self.damp:
            # d_effects = (1. - avg) * torch.sigmoid(-running_activated)
            d_effects = (1. - avg) * (1. - activated)
            h += d_effects
            h = self._relu_norm(h, eps=self.eps)

        return h, activated, running_activated, intermediates

    # def update(self, h, adj_e, adj_i, adj_local):
    #     assert self.edge_update
    #     B, N, K = adj_local.shape
    #



    def subsample_affinities_to_full(self, adj, sample_inds):
        assert sample_inds is not None
        assert sample_inds.shape[0] == 3, "sample_inds should have shape [3, B, N, K]"
        B, N = adj.shape[0:2]
        scatter_adj = torch.zeros([B, N, N]).to(adj)
        scatter_adj = scatter_adj.scatter_(dim=-1, index=sample_inds[2], src=adj)
        return scatter_adj


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
