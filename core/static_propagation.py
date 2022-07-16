import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import pdb
import os
import matplotlib.pyplot as plt
import time
# from torch_sparse import coalesce
from torch_scatter import scatter

class GraphPropagation(nn.Module):
    def __init__(self,
                 num_iters=15,
                 project=True,
                 adj_thresh=0.5,
                 stop_gradient=True,
                 random_init=True,
                 beta=1.0,
                 push=False,
                 damp=False,
                 sharpen=False,
                 inhibit=True,
                 all_propagate=True,
                 normalization_mode='adjacent',
                 subsample_affinity=False):
        super().__init__()
        self.adj_thresh = adj_thresh
        self.stop_gradient = stop_gradient
        self.random_init = random_init
        self.num_iters = num_iters
        self.excite = True
        self.inhibit = inhibit
        self.project = project
        self.beta = beta
        self.push = push
        self.damp = damp
        self.sharpen = sharpen
        self.activated_adj_e = None
        self.activated_adj_i = None
        self.n_senders_e = None
        self.n_senders_i = None
        self.scatter_adj_e = None
        self.scatter_adj_i = None
        self.all_propagate = all_propagate
        self.normalization_mode = normalization_mode
        self.adj_valid = None

    def forward(self, h0, adj, sample_inds=None, activated=None):

        if self.stop_gradient:
            print('warning: stop gradient on h0 and adj')
            h0, adj = h0.clone().detach(), adj.clone().detach()

        # print('warning: stop grad on adj')
        # adj = adj.detach()
        # adj.register_hook(lambda x: print('adj', x.max()))
        B, N, Q = h0.size()

        # split affinities into excitatory and inhibitory
        adj_e, adj_i = adj, 1.0 - adj

        if sample_inds is not None:
            adj_e = self.subsample_affinities_to_full(adj_e, sample_inds)
            adj_i = self.subsample_affinities_to_full(adj_i, sample_inds)

            if self.adj_valid is None:
                self.adj_valid = self.subsample_affinities_to_full(torch.ones_like(adj_e), sample_inds)

        if self.adj_thresh is not None:
            adj_e = self._threshold(adj_e, self.adj_thresh)
            adj_i = self._threshold(adj_i, self.adj_thresh)

        # randomly select the initial index to start propagation

        if activated is None:
            if self.random_init:
                rand_idx = torch.randint(0, N, [B, ]).to(h0.device)
                activated = F.one_hot(rand_idx, num_classes=N).unsqueeze(-1).float()  # [B, N, 1]
            else:
                activated = torch.ones([B, N, 1]).to(adj_e)

        running_activated = activated

        # start propagation
        h = h0.clone()
        h_list = []
        intermediate_list = []

        # print('warning: using experimental version')
        for it in range(self.num_iters):
            # pdb.set_trace()
            h, activated, intermediates = self.propagate_with_full_affinity(h, adj_e, adj_i, activated, running_activated, it)
            # h, activated, intermediates = self.experimental_propagate_with_full_affinity(h, adj_e, adj_i, activated, running_activated, it)
            h_list.append(h)
            intermediate_list.append(intermediates)
        # print(' ----- Total KP time: ', time.time() - start)
        if self.sharpen:
            h = (h * self.beta).softmax(-1)

        return h_list # , intermediate_list

    def propagate_with_full_affinity(self, h, adj_e, adj_i, activated, running_activated, iter):
        # how many sender neurons

        intermediates = {}
        if self.normalization_mode == 'adjacent':
            n_senders_e = torch.sum(adj_e * activated, dim=-2, keepdim=True).clamp(min=1.0).detach() # [B,1,N]
            n_senders_i = torch.sum(adj_i * activated, dim=-2, keepdim=True).clamp(min=1.0).detach() # [B,1,N]
        elif self.normalization_mode == 'activated_local':
            n_senders_e = n_senders_i = torch.sum(self.adj_valid * activated, dim=-2, keepdim=True).clamp(min=1.0).detach() # [B,1,N]
        elif self.normalization_mode == 'constant':
            n_senders_e = n_senders_i = torch.ones_like(activated).permute(0, 2, 1) * 96 ** 0.5
        else:
            raise ValueError

        if self.excite:
            e_effects = torch.matmul(h.permute(0, 2, 1), adj_e * activated) / n_senders_e
            h = h + e_effects.permute(0, 2, 1)

        if self.inhibit:
            i_effects = torch.matmul(h.permute(0, 2, 1), adj_i * activated) / n_senders_i
            i_effects = i_effects.permute(0, 2, 1)

            proj = self._projection(h, i_effects) if self.project else i_effects
            h = h - proj

        h = self._relu_norm(h)
        receivers = torch.max(torch.where(adj_e > adj_i, adj_e, adj_i) * activated, dim=1, keepdim=False)[0] > 0.5 # [B,N]

        if self.random_init:
            running_activated += receivers.unsqueeze(-1).float()  # warning: modify running_activated in place
            activated = running_activated.clamp(max=1.0)
            activated = torch.ones_like(activated) if self.all_propagate else activated

        avg = torch.sum((h * self.beta).softmax(-1) * activated, dim=1, keepdim=True) / (torch.sum(activated, dim=1, keepdim=True) + 1e-6) # [B,1,Q]

        if self.push:
            free = ((1. - avg) * self.beta).softmax(-1)
            p_effects = n_senders_i * activated.permute(0, 2, 1)

            p_effects = p_effects * ((1. - activated) * free).permute(0, 2, 1)
            h += p_effects.permute(0, 2, 1)
            h = self._relu_norm(h)

        if self.damp:
            d_effects = (1. - avg) * torch.sigmoid(-running_activated)
            h += d_effects
            h = self._relu_norm(h)

        return h, activated, intermediates

    def experimental_propagate_with_full_affinity(self, h, adj_e, adj_i, activated, running_activated, iter):
        # how many sender neurons
        # assert self.normalization_mode == 'renormalization'
        intermediates = {}

        A_e = (adj_e * activated).permute(0, 2, 1)
        A_i = (adj_i * activated).permute(0, 2, 1)
        if self.normalization_mode == 'adjacent':
            n_senders_e = torch.sum(adj_e * activated, dim=-2, keepdim=True).clamp(min=1.0).detach().permute(0, 2, 1) # [B,N,1]
            n_senders_i = torch.sum(adj_i * activated, dim=-2, keepdim=True).clamp(min=1.0).detach().permute(0, 2, 1) # [B,N,1]
        elif self.normalization_mode == 'activated_local':
            n_senders_e = n_senders_i = torch.sum(self.adj_valid * activated, dim=-2, keepdim=True).clamp(min=1.0).permute(0, 2, 1).detach() # [B,N,1]
        elif self.normalization_mode == 'constant':
            n_senders_e = n_senders_i = torch.ones_like(activated).permute(0, 2, 1) * 96 ** 0.5
        elif self.normalization_mode == 'renormalization':
            n_senders_e = n_senders_i = torch.ones_like(activated) # [B, N, 1]

            N = adj_e.shape[1]

            eye = torch.eye(N).unsqueeze(0).to(A_e)

            sqrt_deg_e = (A_e + eye).sum(-1, True) ** -0.5
            sqrt_deg_i = (A_i + eye).sum(-1, True) ** -0.5
            diag_deg_e = sqrt_deg_e.expand(-1, -1, N) * eye
            diag_deg_i = sqrt_deg_i.expand(-1, -1, N) * eye
            A_e = diag_deg_e @ A_e @ diag_deg_e
            A_i = diag_deg_i @ A_i @ diag_deg_i
        else:
            raise ValueError

        intermediates['n_sender_e'] = n_senders_e
        intermediates['n_sender_i'] = n_senders_i
        intermediates['h'] = h

        if self.excite:
            e_effects = torch.matmul(A_e, h) / n_senders_e
            h = h + e_effects

        intermediates['e'] = e_effects
        intermediates['h+e'] = h
        if self.inhibit:
            i_effects = torch.matmul(A_i, h) / n_senders_i
            i_effects = i_effects

            proj = self._projection(h, i_effects) if self.project else i_effects
            h = h - proj

        intermediates['i'] = proj
        intermediates['h-i'] = h

        h = self._relu_norm(h)
        intermediates['norm'] = h
        receivers = torch.max(torch.where(adj_e > adj_i, adj_e, adj_i) * activated, dim=1, keepdim=False)[0] > 0.5 # [B,N]

        if self.random_init:
            running_activated = running_activated + receivers.unsqueeze(-1).float()
            activated = running_activated.clamp(max=1.0)
            activated = torch.ones_like(activated) if self.all_propagate else activated

        avg = torch.sum((h * self.beta).softmax(-1) * activated, dim=1, keepdim=True) / (torch.sum(activated, dim=1, keepdim=True) + 1e-6) # [B,1,Q]

        if self.push:
            free = ((1. - avg) * self.beta).softmax(-1)
            p_effects = n_senders_i * activated.permute(0, 2, 1)

            p_effects = p_effects * ((1. - activated) * free).permute(0, 2, 1)
            h += p_effects.permute(0, 2, 1)
            h = self._relu_norm(h)

        if self.damp:
            d_effects = (1. - avg) * torch.sigmoid(-running_activated)
            h += d_effects
            h = self._relu_norm(h)

        return h, activated, intermediates

    def subsample_affinities_to_full(self, adj, sample_inds):
        assert sample_inds is not None
        assert sample_inds.shape[0] == 3, "sample_inds should have shape [3, B, N, K]"
        B, N = adj.shape[0:2]
        scatter_adj = torch.zeros([B, N, N+1]).to(adj)
        scatter_adj = scatter_adj.scatter_(dim=-1, index=sample_inds[2], src=adj)
        return scatter_adj[:, :, 0:N]


    @staticmethod
    def _threshold(x, thresh):
        return x * (x > thresh).float()

    @staticmethod
    def _projection(v, u, eps=1e-12):
        u_norm = torch.sum(u * u, -1, keepdims=True)
        dot_prod = torch.sum(v * u, -1, keepdims=True)
        proj = (dot_prod / (u_norm + eps)) * u
        return proj

    @staticmethod
    def _relu_norm(x):
        return F.normalize(F.relu(x), p=2.0, dim=-1, eps=1e-8)


def object_id_hash(objects, dtype_out=torch.int32, val=256, channels_last=False):
    '''
    objects: [...,C]
    val: a number castable to dtype_out

    returns:
    out: [...,1] where each value is given by sum([val**(C-1-c) * objects[...,c:c+1] for c in range(C)])
    '''
    if not isinstance(objects, torch.Tensor):
        objects = torch.tensor(objects)
    if not channels_last:
        objects = objects.permute(0, 2, 3, 1)
    C = objects.shape[-1]
    val = torch.tensor(val, dtype=dtype_out)
    objects = objects.to(dtype_out)
    out = torch.zeros_like(objects[..., 0:1])
    for c in range(C):
        scale = torch.pow(val, C - 1 - c)
        out += scale * objects[..., c:c + 1]
    if not channels_last:
        out = out.permute(0, 3, 1, 2)

    return out


def create_local_ind_buffer(img_size, K):
    H, W = img_size
    indice_maps = torch.arange(H * W).reshape([1, 1, H, W]).float()

    # symmetric padding
    assert K % 2 == 1  # assert K is odd
    half_K = int((K - 1) / 2)

    pad_fn = nn.ReflectionPad2d(half_K)
    indice_maps = pad_fn(indice_maps)
    local_inds = F.unfold(indice_maps, kernel_size=K, stride=1)  # [B, C * K * k, H, W]
    local_inds = local_inds.permute(0, 2, 1)
    return local_inds


def compute_gt_affinity(objects, channels_last, subsample=False, K=11, S=512):
    objects = object_id_hash(objects, channels_last=channels_last) # [B, H, W, 1]

    if channels_last:
        B, H, W, _ = objects.shape
    else:
        B, _, H, W = objects.shape

    N = H * W
    objects = objects.reshape(B, H * W)
    gt_affinity = objects.unsqueeze(-1) == objects.unsqueeze(-2)

    if subsample:
        # local indices
        local_inds = create_local_ind_buffer([H, W], K=K)
        local_inds = local_inds.expand(B, -1, -1)  # tile local indices in batch dimension

        # random global indices

        device = local_inds.device
        if S > K ** 2:
            rand_global_inds = torch.randint(H * W, [B, H * W, S-K**2], device=device)
            sample_inds = torch.cat([local_inds, rand_global_inds], -1).long()

        else:
            S = K ** 2
            sample_inds = local_inds

        sample_inds = sample_inds.reshape([1, B, N, S])
        # concatenate with batch and node indices
        batch_inds = torch.arange(B, device=device).reshape([1, B, 1, 1]).expand(-1, -1, N, S)
        node_inds = torch.arange(N, device=device).reshape([1, 1, N, 1]).expand(-1, B, -1, S)
        sample_inds = torch.cat([batch_inds, node_inds, sample_inds], 0).long()  # [3, B, N, S]
        gt_affinity = gt_affinity[list(sample_inds)]
    else:
        sample_inds = None
    return objects, gt_affinity.float(), sample_inds


def test_gt_affinity(num_test_images=5, subsample=False):
    from PIL import Image

    # instantiate competition module
    graph_propagation = GraphPropagation()
    H, W = 32, 32
    # load GT images
    folder_path = '/mnt/fs6/honglinc/dataset/playroom_large_v1_main_images/objects/model_split_0'
    for i in range(num_test_images):
        image_path = os.path.join(folder_path, format(i, '04d'), format(5, '05d') + '.png')
        image = Image.open(image_path)
        image = np.asarray(image)

        image = F.interpolate(torch.tensor(image[None]).permute(0, 3, 1, 2).float().cuda(), size=[H, W], mode='nearest').long()

        start = time.time()
        objects, gt_affinity, sample_inds = compute_gt_affinity(image, channels_last=False, subsample=subsample)
        print('Forward time: ', time.time() - start)

        B, N, Q = 1, H * W, 256
        h0 = torch.randn([B, N, Q]).softmax(-1).cuda()

        h = graph_propagation.forward(h0, gt_affinity, sample_inds)
        labels = h.argmax(-1).reshape(H, W)
        _, segment_ids = torch.unique(labels, return_inverse=True)
        segment_ids -= segment_ids.min()

        # # --- Visualization ---
        pdb.set_trace()

        plt.subplot(1, 2, 1)
        plt.imshow(image[0].permute(1, 2, 0).cpu())
        plt.title('GT Segments: %d' % len(torch.unique(objects)))
        plt.subplot(1, 2, 2)
        plt.imshow(segment_ids.cpu())
        plt.title('Output segment: %d' % len(torch.unique(segment_ids)))
        plt.show()
        plt.close()


if __name__ == '__main__':
    test_gt_affinity(1, subsample=True)
    # test_gt_affinity(50)
    # B, N, Q = 2, 1024, 32
    #
    # H0 = torch.randn([B, N, Q]).softmax(-1).cuda()
    # A = torch.randn([B, N, N]).softmax(-1).cuda()
    # A /= A.max(-1)[0].unsqueeze(-1)
    #
    # graph_propagation = GraphPropagation()
    # h = graph_propagation.forward(H0, A)
    # print(h, h)
