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
                 latent_dim=64,
                 subsample_affinity=True,
                 eval_full_affinity=False,
                 local_window_size=25,
                 num_affinity_samples=1024,
                 propagation_iters=25,
                 propagation_affinity_thresh=0.5,
                 num_masks=32,
                 num_competition_rounds=2,
                 stem_pool=True,
                 flow_threshold=0.5,
    ):
        super(EISEN, self).__init__()

        # [Backbone encoder]
        self.backbone = ResNet_Deeplab()
        output_dim = self.backbone.output_dim

        # [Affinity decoder]
        self.feat_conv = nn.Conv2d(output_dim, kq_dim, kernel_size=1, bias=True, padding='same')
        self.key_proj = nn.Linear(kq_dim, kq_dim)
        self.query_proj = nn.Linear(kq_dim, kq_dim)

        # [Affinity sampling]
        self.sample_affinity = subsample_affinity and (not (eval_full_affinity and (not self.training)))
        self.register_buffer('local_indices', utils.generate_local_indices(img_size=affinity_res, K=local_window_size))

        # [Propagation]
        self.propagation = GraphPropagation(
            num_iters=propagation_iters, adj_thresh=propagation_affinity_thresh, stop_gradient=False, push=False, damp=False)

        # [Competition]
        self.competition = Competition(num_masks=num_masks, num_competition_rounds=num_competition_rounds)

        self.local_window_size = local_window_size
        self.num_affinity_samples = num_affinity_samples
        self.affinity_res = affinity_res
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1), False)

    def forward(self, input, segment_target, get_segments=False):
        """ build outputs at multiple levels"""
        # [Normalize inputs]
        img = (input['img1'] - self.pixel_mean) / self.pixel_std

        # [Backbone
        features = self.backbone(img)
        # Key & query projection
        features = self.feat_conv(features).permute(0, 2, 3, 1) # [B, H, W, C]
        key = self.key_proj(features).permute(0, 3, 1, 2) # [B, C, H, W]
        query = self.query_proj(features).permute(0, 3, 1, 2) # [B, C, H, W]
        B, C, H, W = key.shape

        # Sampling affinities
        sample_inds = self.generate_affinity_sample_indices(size=[B, H, W]) if self.sample_affinity else None

        # Compute affinity logits
        affinity_logits = self.compute_affinity_logits(key, query, sample_inds)
        affinity_logits *= C ** -0.5

        # Compute affinity loss
        loss = self.compute_loss(affinity_logits, sample_inds, segment_target)

        if get_segments:
            # Compute segments via propagation and competition
            segments = self.compute_segments(affinity_logits, sample_inds)
            # Compute segment metrics
            gt_segment = input['gt_segment']
            seg_metric = {'metric_pred_segment_mean_ious': loss}

            # seg_metric, seg_out = self.measure_segments(segments,  gt_segment)
            # seg_metric = {k: v.to(loss.device) for k, v in seg_metric.items()} # tensor needs to be on GPU
            # self.visualize_segments_all(segments, gt_segment[None], torch.zeros_like(gt_segment), input['img1'] / 255.)

            return affinity_logits, loss, seg_metric, None
        else:
            return affinity_logits, loss, None, None


    def generate_affinity_sample_indices(self, size):
        # local_indices and local_masks below are stored in the buffers
        # so that we don't have to do the same computation at every iteration

        B, H, W = size
        S = self.num_affinity_samples
        K = self.local_window_size
        local_inds = self.local_indices.expand(B, -1, -1)  # tile local indices in batch dimension
        device = local_inds.device

        if K ** 2 <= S:
            # compute random global indices
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

    def compute_loss(self, logits, sample_inds, segment_targets):
        """ build loss for a single level """

        B, N, K = logits.shape

        # 0. compute binary affinity targets
        segment_targets = F.interpolate(segment_targets.float(), self.affinity_res, mode='nearest')
        segment_targets = segment_targets.reshape(B, N).unsqueeze(-1).long()
        if sample_inds is not None:
            samples = utils.gather_tensor(segment_targets, sample_inds[[0, 2]]).squeeze(-1)
        else:
            samples = segment_targets.permute(0, 2, 1)
        targets = segment_targets == samples
        null_mask = (segment_targets == 0) # & (samples == 0)  only mask the rows
        mask = 1 - null_mask.float()

        # 1. compute log softmax on the logits (F.kl_div requires log prob for pred)
        y_pred = utils.weighted_softmax(logits, mask)
        y_pred = torch.log(y_pred.clamp(min=1e-8))  # log softmax

        # 2. compute the target probabilities (F.kl_div requires prob for target)
        y_true = targets / (torch.sum(targets, -1, keepdim=True) + 1e-9)

        # 3. compute kl divergence
        kl_div = F.kl_div(y_pred, y_true, reduction='none') * mask
        kl_div = kl_div.sum(-1)

        # 4. average kl divergence aross rows with non-empty positive / negative labels
        agg_mask = (mask.sum(-1) > 0).float()
        loss = kl_div.sum() / (agg_mask.sum() + 1e-9)


        return loss

    def compute_segments(self, logits, sample_inds, run_cc=True, hidden_dim=32):
        B, N, K = logits.shape

        h0 = torch.cuda.FloatTensor(B, N, hidden_dim).normal_().softmax(-1) # h0
        adj = utils.softmax_max_norm(logits) # normalize affinities

        # Convert affinity matrix to sparse tensor for memory efficiency
        adj = utils.local_to_sparse_global_affinity(adj, sample_inds, sparse_transpose=True) # sparse affinity matrix

        # KP
        hidden_states, _, _ = self.propagation(h0.detach(), adj.detach())
        plateau = hidden_states[-1].reshape([B, self.affinity_res[0], self.affinity_res[1], hidden_dim])

        # Competition
        masks, agents, alive, phenotypes, unharvested = self.competition(plateau)
        instance_seg = masks.argmax(-1)

        if not run_cc:
            segments = instance_seg
        else: # Connected component
            cc_labels = [label_connected_component(instance_seg[i], min_area=10)[None]
                         for i in range(B)] #[[1, H, W]]
            segments = torch.cat(cc_labels)

        return segments

    def measure_segments(self, pred_segment, gt_segment):
        return measure_static_segmentation_metric({'pred_segment': pred_segment}, {'gt_segment': gt_segment},
                                                  pred_segment.shape[-2:],
                                                  segment_key=['pred_segment'],
                                                  moving_only=False,
                                                  eval_full_res=True)

    def visualize_segments_all(self, pred_segment, gt_segment, target, image, prefix=''):

        H = W = 64

        fsz = 19
        num_plots = 4
        fig = plt.figure(figsize=(num_plots * 4, 5))
        gs = fig.add_gridspec(1, num_plots)
        ax1 = fig.add_subplot(gs[0])

        plt.imshow(image[0].permute([1, 2, 0]).cpu())
        plt.axis('off')
        ax1.set_title('Image', fontsize=fsz)

        # labels = F.interpolate(batched_inputs[0]['gt_moving'].unsqueeze(0).float().cuda(), size=[H, W], mode='nearest')
        ax = fig.add_subplot(gs[1])

        if target is None:
            target = torch.zeros(1, 1, H, W)
        plt.imshow(target[0].cpu())
        plt.title('Supervision', fontsize=fsz)
        plt.axis('off')

        ax = fig.add_subplot(gs[2])
        plt.imshow(pred_segment[0].cpu(), cmap='twilight')
        plt.title('Pred segments', fontsize=fsz)
        plt.axis('off')

        ax = fig.add_subplot(gs[3])
        plt.imshow(gt_segment[0, 0].cpu(), cmap='twilight')
        plt.title('GT segments', fontsize=fsz)
        plt.axis('off')

        # file_idx = batched_inputs[0]['file_name'].split('/')[-1].split('.hdf5')[0]
        # save_path = os.path.join(self.vis_saved_path, 'step_%smask_%s_%s.png' % (prefix, 'eval' if iter is None else str(iter), file_idx))
        # print('Save fig to ', save_path)
        # plt.savefig(save_path, bbox_inches='tight')

        plt.show()
        plt.close()
    def load_checkpoint(self):

        ckpt_path = '/data3/honglinc/detectron_exp_log/TDW_128_RAFT_sintel0.5_b8_simple_aspp_nopos_nosym/model_0009999.pth'
        state_dict = torch.load(ckpt_path)['model']
        keys_list = list(state_dict.keys())
        # breakpoint()
        #
        # for k,v in state_dict.items():
        #     print(k, v.shape)
        # print('------' * 10)
        new_state_dict = self.state_dict()

        #
        # for k,v in new_state_dict.items():
        #     print(k, v.shape)
        # breakpoint()
        for k, v in new_state_dict.items():
            if 'backbone' in k:
                new_k = 'net.student_encoders.f_single.' + k[9:]
            elif 'key_proj' in k:
                new_k = k.replace('key_proj', 'net.student_decoders.spatial_decoder.key')
            elif 'query_proj' in k:
                new_k = k.replace('query_proj', 'net.student_decoders.spatial_decoder.query')
            elif 'feat_conv' in k:
                new_k = k.replace('feat_conv', 'net.student_decoders.spatial_decoder.conv')
                # if 'conv1' in k:
                #     new_k = new_k.replace('conv1', 'blocks.0.blocks.0.0')
                # elif 'norm1' in k:
                #     new_k = new_k.replace('norm1', 'blocks.0.blocks.0.1')
                # elif 'conv2' in k:
                #     new_k = new_k.replace('conv2', 'blocks.0.blocks.2.0')
                # elif 'norm2' in k:
                #     new_k = new_k.replace('norm2', 'blocks.0.blocks.2.1')
                # elif 'shortcut' in k:
                #     new_k = new_k.replace('shortcut', 'blocks.0.shortcut')

            else:
                print('Skipping others: ', k)
                continue

            if new_k not in state_dict.keys():
                print('Skipping unfound: ', k)

                continue

            new_state_dict[k] = state_dict[new_k]
            print('*', k, '-->', new_k)
            assert new_k in keys_list, pdb.set_trace()
            keys_list.remove(new_k)


        for k in keys_list:
            print(k)

        self.load_state_dict(new_state_dict)

if __name__ == "__main__":
    x = torch.randn([1, 3, 512, 512]).cuda()
    segment_target = (torch.randn([1, 512, 512]) > 0.5).cuda().long()
    model = EISEN().cuda()
    for i in range(10):
        model(x, segment_target, get_segment=True)