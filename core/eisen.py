# Copyright (c) Facebook, Inc. and its affiliates.
import pdb

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# from resnet import KeyQueryNetwork
from core.resnet import ResNetFPN, ResNet_Deeplab
from core.projection import KeyQueryProjection

from core.competition import Competition
from core.utils.connected_component import label_connected_component
from core.utils.segmentation_metrics import measure_static_segmentation_metric
import core.utils.utils as utils
import matplotlib.pyplot as plt
from core.propagation import GraphPropagation
import time

class EISEN(nn.Module):
    def __init__(self,
                 affinity_res=[128, 128],
                 kq_dim=32,
                 subsample_affinity=True,
                 eval_full_affinity=False,
                 local_window_size=25,
                 num_affinity_samples=1024,
                 propagation_iters=25,
                 propagation_affinity_thresh=0.7,
                 num_masks=32,
                 num_competition_rounds=3,
                 supervision_level=3,
    ):
        super(EISEN, self).__init__()

        self.local_window_size = local_window_size
        self.num_affinity_samples = num_affinity_samples
        self.affinity_res = affinity_res
        self.supervision_level = supervision_level

        # [Backbone encoder]
        self.backbone = ResNet_Deeplab()
        output_dim = self.backbone.output_dim

        # [Affinity decoder]
        self.feat_conv = nn.Conv2d(output_dim, kq_dim, kernel_size=1, bias=True, padding='same')
        self.key_proj = nn.Linear(kq_dim, kq_dim)
        self.query_proj = nn.Linear(kq_dim, kq_dim)

        # [Affinity sampling]
        self.sample_affinity = subsample_affinity and (not (eval_full_affinity and (not self.training)))
        for level in range(supervision_level):
            stride = 2 ** level
            H, W = affinity_res[0]//stride, affinity_res[1]//stride
            buffer_name = f'local_indices_{H}_{W}'
            self.register_buffer(buffer_name, utils.generate_local_indices(img_size=[H, W], K=local_window_size).cuda(), persistent=False)

        # [Propagation]
        self.propagation = GraphPropagation(num_iters=propagation_iters, adj_thresh=propagation_affinity_thresh)

        # [Competition]
        self.competition = Competition(num_masks=num_masks, num_competition_rounds=num_competition_rounds)

        # [Input normalization]
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1), False)

    def forward(self, input, segment_target, get_segments=False, vis_segments=False):
        """ build outputs at multiple levels"""
        # [Normalize inputs]
        img = (input['img1'] - self.pixel_mean) / self.pixel_std

        # [Backbone]
        features = self.backbone(img)

        # [Key & query projection]
        features = self.feat_conv(features).permute(0, 2, 3, 1) # [B, H, W, C]
        key = self.key_proj(features).permute(0, 3, 1, 2) # [B, C, H, W]
        query = self.query_proj(features).permute(0, 3, 1, 2) # [B, C, H, W]
        B, C, H, W = key.shape

        loss_list = []
        affinity_list = []
        sample_inds = None

        # Compute affinity loss at multiple scales
        for level in range(self.supervision_level if self.training else 1):
            stride = 2 ** level

            # [Sampling affinities]
            if self.sample_affinity:
                sample_inds = self.generate_affinity_sample_indices(size=[B, H//stride, W//stride])

            # [Compute affinity logits]
            affinity_logits = self.compute_affinity_logits(
                key=utils.downsample_tensor(key, stride),
                query=utils.downsample_tensor(query, stride),
                sample_inds=sample_inds
            ) * (C ** -0.5)
            affinity_list.append(affinity_logits)

            # [Compute affinity loss]
            loss = self.compute_loss(affinity_logits, sample_inds, segment_target, [H//stride, W//stride])
            loss_list.append(loss.unsqueeze(0))

        loss = sum(loss_list)

        # [Compute segmentation masks]
        if get_segments:
            # Compute segments via propagation and competition
            segments = self.compute_segments(affinity_list[0], sample_inds)
            # Compute segment metrics
            gt_segment = input['gt_segment']
            seg_metric, seg_out = self.measure_segments(segments,  gt_segment)
            seg_metric = {k: v.to(loss.device).unsqueeze(0) for k, v in seg_metric.items()} 

            if vis_segments:
                self.visualize_segments(seg_out['pred_segment'], input)

            return affinity_list, loss, seg_metric, None
        else:
            return affinity_list, loss, None, None


    def generate_affinity_sample_indices(self, size):
        B, H, W = size
        S = self.num_affinity_samples
        K = self.local_window_size
        # local_indices and local_masks below are stored in the buffers
        # so that we don't have to repeat the same computation at every iteration
        local_inds = getattr(self, f'local_indices_{H}_{W}').expand(B, -1, -1)  # tile local indices in batch dimension
        device = local_inds.device

        if K ** 2 <= S:
            # sample random global indices
            rand_global_inds = torch.randint(H * W, [B, H*W, S-K**2], device=device)
            sample_inds = torch.cat([local_inds, rand_global_inds], -1)
        else:
            sample_inds = local_inds

        # create gather indices
        sample_inds = sample_inds.reshape([1, B, H*W, S])
        batch_inds = torch.arange(B, device=device).reshape([1, B, 1, 1]).expand(-1, -1, H*W, S)
        node_inds = torch.arange(H*W, device=device).reshape([1, 1, H*W, 1]).expand(-1, B, -1, S)
        sample_inds = torch.cat([batch_inds, node_inds, sample_inds], 0).long()  # [3, B, N, S]

        return sample_inds

    def compute_affinity_logits(self, key, query, sample_inds):
        B, C, H, W = key.shape
        key = key.reshape([B, C, H * W]).permute(0, 2, 1)      # [B, N, C]
        query = query.reshape([B, C, H * W]).permute(0, 2, 1)  # [B, N, C]

        if self.sample_affinity: # subsample affinity
            gathered_query = utils.gather_tensor(query, sample_inds[[0, 1], ...])
            gathered_key = utils.gather_tensor(key, sample_inds[[0, 2], ...])
            logits = (gathered_query * gathered_key).sum(-1)  # [B, N, K]
        else: # full affinity
            logits = torch.matmul(query, key.permute(0, 2, 1))

        return logits

    def compute_loss(self, logits, sample_inds, segment_targets, size):
        B, N, K = logits.shape

        # [compute binary affinity targets]
        segment_targets = F.interpolate(segment_targets.float(), size, mode='nearest')
        segment_targets = segment_targets.reshape(B, N).unsqueeze(-1).long()
        if sample_inds is not None:
            samples = utils.gather_tensor(segment_targets, sample_inds[[0, 2]]).squeeze(-1)
        else:
            samples = segment_targets.permute(0, 2, 1)
        targets = segment_targets == samples
        null_mask = (segment_targets == 0) # & (samples == 0)  only mask the rows
        mask = 1 - null_mask.float()

        # [compute log softmax on the logits] (F.kl_div requires log prob for pred)
        y_pred = utils.weighted_softmax(logits, mask)
        y_pred = torch.log(y_pred.clamp(min=1e-8))  # log softmax

        # [compute the target probabilities] (F.kl_div requires prob for target)
        y_true = targets / (torch.sum(targets, -1, keepdim=True) + 1e-9)

        # [compute kl divergence]
        kl_div = F.kl_div(y_pred, y_true, reduction='none') * mask
        kl_div = kl_div.sum(-1)

        # [average kl divergence aross rows with non-empty positive / negative labels]
        agg_mask = (mask.sum(-1) > 0).float()
        loss = kl_div.sum() / (agg_mask.sum() + 1e-9)

        return loss

    def compute_segments(self, logits, sample_inds, hidden_dim=32, run_cc=True, min_cc_area=20):
        B, N, K = logits.shape

        # [Initialize hidden states]
        h0 = torch.cuda.FloatTensor(B, N, hidden_dim).normal_().softmax(-1) # h0

        # [Process affinities]
        adj = utils.softmax_max_norm(logits) # normalize affinities
        # Convert affinity matrix to sparse tensor for memory efficiency, if subsample_affinity = True
        adj = utils.local_to_sparse_global_affinity(adj, sample_inds, sparse_transpose=True) # sparse affinity matrix

        # [Graph propagation]
        plateau_map_list = self.propagation(h0.detach(), adj.detach())
        plateau_map = plateau_map_list[-1].reshape([B, self.affinity_res[0], self.affinity_res[1], hidden_dim])

        # [Competition]
        masks, agents, alive, phenotypes, unharvested = self.competition(plateau_map)
        instance_seg = masks.argmax(-1)

        # [Run connected component]
        if not run_cc:
            segments = instance_seg
        else: # apply connected component algorithm
            cc_labels = [label_connected_component(instance_seg[i], min_area=min_cc_area)[None]
                         for i in range(B)] #[[1, H, W]]
            segments = torch.cat(cc_labels)

        return segments

    def measure_segments(self, pred_segment, gt_segment):
        return measure_static_segmentation_metric({'pred_segment': pred_segment}, {'gt_segment': gt_segment},
                                                  pred_segment.shape[-2:],
                                                  segment_key=['pred_segment'],
                                                  eval_full_res=True)

    def visualize_segments(self, seg_out, input):
        num_objects = len(seg_out[0][0])
        pred_obj_seg, gt_obj_seg, iou = seg_out
        fig, axs = plt.subplots(2, 1 + num_objects, figsize=(10, 4))
        axs[0, 0].imshow(input['img1'][0].permute(1, 2, 0).cpu())
        axs[0, 0].set_title('Input image', fontsize=10)
        axs[1, 0].imshow(input['gt_segment'][0].cpu())
        axs[1, 0].set_title('GT segments', fontsize=10)

        for n in range(num_objects):
            axs[0, n + 1].imshow(pred_obj_seg[0][n])
            axs[0, n + 1].set_title(f'Pred. segment {n}', fontsize=10)
            axs[1, n + 1].imshow(gt_obj_seg[0][n])
            axs[1, n + 1].set_title(f'GT segment {n}', fontsize=10)

        for i in range(axs.shape[-1]):
            axs[0, i].set_axis_off()
            axs[1, i].set_axis_off()
        plt.show()
        plt.close()


if __name__ == "__main__":
    x = torch.randn([1, 3, 512, 512]).cuda()
    segment_target = (torch.randn([1, 512, 512]) > 0.5).cuda().long()
    model = EISEN().cuda()
    for i in range(10):
        model(x, segment_target, get_segment=True)