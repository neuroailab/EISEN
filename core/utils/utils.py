import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from torch_sparse import SparseTensor

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def gather_tensor(tensor, sample_inds, invalid=0.):
    # tensor is of shape [B, N, D]
    # sample_inds is of shape [2, B, T, K] or [3, B, T, K]
    # where the last column of the 1st dimension are the sample indices

    _, N, D = tensor.shape
    dim, B, T, K = sample_inds.shape


    if dim == 2:
        if sample_inds[-1].max() == N:
            # special case: indices where idx == N is assigned zero
            tensor = torch.cat([tensor, invalid * torch.ones([B, 1, D], device=tensor.device)], dim=1)

        indices = sample_inds[-1].view(B, T * K).unsqueeze(-1).expand(-1, -1, D)
        output = torch.gather(tensor, 1, indices).view([B, T, K, D])
    elif dim == 3:
        if sample_inds[-1].max() == D:
            # special case: indices where idx == N is assigned zero
            tensor = torch.cat([tensor, invalid * torch.ones([B, N, 1], device=tensor.device)], dim=2)
            D = D + 1
        elif sample_inds[1].max() == N:
            # special case: indices where idx == N is assigned zero
            tensor = torch.cat([tensor, invalid * torch.ones([B, 1, D], device=tensor.device)], dim=1)
            N = N + 1

        tensor = tensor.view(B, N * D)
        node_indices = sample_inds[1].view(B, T * K)
        sample_indices = sample_inds[2].view(B, T * K)
        indices = node_indices * D + sample_indices
        # print('in gather tensor: ', indices.max(), tensor.shape)
        output = torch.gather(tensor, 1, indices).view([B, T, K])
    else:
        raise ValueError
    return output

def softmax_max_norm(x):
    x = x.softmax(-1)
    x = x / torch.max(x, dim=-1, keepdim=True)[0].clamp(min=1e-12)# .detach()
    return x

def weighted_softmax(x, weight):
    maxes = torch.max(x, -1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = (torch.sum(x_exp * weight, -1, keepdim=True) + 1e-12)
    return (x_exp / x_exp_sum) * weight

def reorder_int_labels(x):
    _, y = torch.unique(x, return_inverse=True)
    y -= y.min()
    return y


def generate_local_indices(img_size, K, padding='reflection'):
    H, W = img_size
    indice_maps = torch.arange(H * W).reshape([1, 1, H, W]).float()

    # symmetric padding
    assert K % 2 == 1  # assert K is odd
    half_K = int((K - 1) / 2)

    assert padding in ['reflection', 'constant'], "unsupported padding mode"
    if padding == 'reflection':
        pad_fn = torch.nn.ReflectionPad2d(half_K)
    else:
        pad_fn = torch.nn.ConstantPad2d(half_K, H * W)

    indice_maps = pad_fn(indice_maps)
    local_inds = F.unfold(indice_maps, kernel_size=K, stride=1)  # [B, C * K * k, H, W]
    local_inds = local_inds.permute(0, 2, 1)
    return local_inds


def local_to_sparse_global_affinity(local_adj, sample_inds, activated=None, sparse_transpose=False):
    """
    Convert local adjacency matrix of shape [B, N, K] to [B, N, N]
    :param local_adj: [B, N, K]
    :param size: [H, W], with H * W = N
    :return: global_adj [B, N, N]
    """

    B, N, K = list(local_adj.shape)

    assert sample_inds.shape[0] == 3
    local_node_inds = sample_inds[2] # [B, N, K]

    batch_inds = torch.arange(B).reshape([B, 1]).to(local_node_inds)
    node_inds = torch.arange(N).reshape([1, N]).to(local_node_inds)
    row_inds = (batch_inds * N + node_inds).reshape(B * N, 1).expand(-1, K).flatten()  # [BNK]

    col_inds = local_node_inds.flatten()  # [BNK]
    valid = col_inds < N

    col_offset = (batch_inds * N).reshape(B, 1, 1).expand(-1, N, -1).expand(-1, -1, K).flatten() # [BNK]
    col_inds += col_offset
    value = local_adj.flatten()

    if activated is not None:
        activated = activated.reshape(B, N, 1).expand(-1, -1, K).bool()
        valid = torch.logical_and(valid, activated.flatten())

    if sparse_transpose:
        global_adj = SparseTensor(row=col_inds[valid], col=row_inds[valid],
                                  value=value[valid], sparse_sizes=[B*N, B*N])
    else:
        raise ValueError('Current KP implementation assumes tranposed affinities')

    return global_adj


def kl_divergence(logits, labels, logits_mode='log_softmax', labels_mode='row_sum', reduce_sum=True, label_smoothing=False):
    """
    :param logits: [B, N, K] raw logits (pre-softmax)
    :param labels: [B, N, K] binary labels
    :return: [B, N] KL divergence
    """
    B, N, K = logits.shape

    logits = logits.reshape([B * N, K])
    labels = labels.reshape([B * N, K])

    if logits_mode == 'log_softmax':
        logits = F.log_softmax(logits, -1)       # log probabilities
    elif logits_mode == 'log':
        logits = logits.clamp(min=1e-9).log()
    elif logits_mode == 'log_row_norm':
        logits = logits / (logits.sum(-1, keepdim=True) + 1e-9)   #  probabilities
        logits = logits.clamp(min=1e-9).log()
    else:
        raise ValueError

    if labels_mode == 'softmax':
        labels = torch.softmax(labels, -1)
    else:
        #speedup change
        labels = labels / (labels.sum(-1, keepdim=True) + 1e-9)  #  probabilities

    if label_smoothing:
        assert not labels_mode == 'softmax'
        alpha = 0.1
        labels = (1 - alpha) * labels + alpha / labels.shape[-1]

    kl_div = F.kl_div(logits, labels, reduction='none')

    return kl_div.sum(-1) if reduce_sum else kl_div
