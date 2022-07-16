import numpy as np
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import pdb, os
import matplotlib.pyplot as plt
from kornia.filters.kernels import (get_spatial_gradient_kernel2d,
                                    normalize_kernel2d)
import time
import sys

# from .panoptic_seg import kl_divergence

def l2_normalize(x):
    return F.normalize(x, p=2.0, dim=-1, eps=1e-6)


def reduce_max(x, dim, keepdim=True):
    return torch.max(x, dim=dim, keepdim=keepdim)[0]


def coordinate_ims(batch_size, seq_length, imsize):
    static = False
    if seq_length == 0:
        static = True
        seq_length = 1
    B = batch_size
    T = seq_length
    H, W = imsize
    ones = torch.ones([B, H, W, 1], dtype=torch.float32).cuda()
    h = torch.divide(torch.arange(H).to(ones), torch.tensor(H - 1, dtype=torch.float32).to(ones))
    h = 2.0 * ((h.view(1, H, 1, 1) * ones) - 0.5)
    w = torch.divide(torch.arange(W).to(ones), torch.tensor(W - 1, dtype=torch.float32).to(ones))
    w = 2.0 * ((w.view(1, 1, W, 1) * ones) - 0.5)
    h = torch.stack([h] * T, 1)
    w = torch.stack([w] * T, 1)
    hw_ims = torch.cat([h, w], -1)
    if static:
        hw_ims = hw_ims[:, 0]
    return hw_ims


def dot_product_attention(queries, keys, normalize=True, eps=1e-8):
    """
    Compute the normalized dot product between two PyTorch tensors
    """
    B, N, D_q = queries.size()
    _B, N_k, D_k = keys.size()
    assert D_q == D_k, (queries.shape, keys.shape)
    if normalize:
        queries = F.normalize(queries, p=2.0, dim=-1, eps=eps)
        keys = F.normalize(keys, p=2.0, dim=-1, eps=eps)

    outputs = torch.matmul(queries, torch.transpose(keys, 1, 2))  # [B, N, N_k]
    attention = torch.transpose(outputs, 1, 2)  # [B, N_k, N]

    return outputs


def sample_image_inds_from_probs(probs, num_points, eps=1e-8):
    B, H, W = probs.shape
    P = num_points
    N = H * W

    probs = probs.reshape(B, N)

    probs = torch.maximum(probs + eps, torch.tensor(0.).to(probs)) / (probs.sum(dim=-1, keepdim=True) + eps)
    dist = Categorical(probs=probs)
    indices = dist.sample([P]).permute(1, 0).to(torch.int32)  # [B,P]

    # indices_h = torch.minimum(torch.maximum(torch.div(indices, W, rounding_mode='floor'), torch.tensor(0)), torch.tensor(H-1))
    indices_h = torch.minimum(torch.maximum(indices // W, torch.tensor(0).to(indices)), torch.tensor(H - 1).to(indices))
    indices_w = torch.minimum(torch.maximum(torch.fmod(indices, W), torch.tensor(0).to(indices)), torch.tensor(W - 1).to(indices))
    indices = torch.stack([indices_h, indices_w], dim=-1)  # [B,P,2]
    return indices


def get_gradient_image(image, mode='sobel', order=1, normalize_kernel=True):
    B, C, H, W = list(image.size())

    # prepare kernel
    kernel = get_spatial_gradient_kernel2d(mode, order)
    if normalize_kernel:
        kernel = normalize_kernel2d(kernel)
    tmp_kernel = kernel.to(image).detach()
    tmp_kernel = tmp_kernel.unsqueeze(1).unsqueeze(1)
    kernel_flip = tmp_kernel.flip(-3)

    # pad spatial dims of image
    padding = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels = 3 if (order == 2) else 2
    padded_image = F.pad(image.reshape(B * C, 1, H, W), padding, 'replicate')[:, :, None]  # [B*C,1,1,H+p,W+p]
    gradient_image = F.conv3d(padded_image, kernel_flip, padding=0).view(B, C, out_channels, H, W)
    return gradient_image


def sample_coordinates_at_borders(image, num_points=16, mask=None, sum_edges=True, normalized_coordinates=True):
    """
    Sample num_points in normalized (h,w) coordinates from the borders of the input image
    """
    B, C, H, W = list(image.size())
    if mask is not None:
        assert mask.shape[2:] == image.shape[2:], (mask.size(), image.size())
    else:
        mask = torch.ones(size=(B, 1, H, W)).to(image)

    gradient_image = get_gradient_image(image * mask, mode='sobel', order=1)  # [B,C,2,H,W]
    gradient_magnitude = torch.sqrt(torch.square(gradient_image).sum(dim=2))
    if sum_edges:
        edges = gradient_magnitude.sum(1)  # [B,H,W]
    else:
        edges = gradient_magnitude.max(1)[0]

    if mask is not None:
        edges = edges * mask[:, 0]

    coordinates = sample_image_inds_from_probs(edges, num_points=num_points)
    if normalized_coordinates:
        coordinates = coordinates.float()
        coordinates /= torch.tensor([H - 1, W - 1], dtype=torch.float32).to(coordinates)[None, None]
        coordinates = 2.0 * coordinates - 1.0
    return coordinates


def index_into_images(images, indices, channels_last=False):
    """
    index into an image at P points to get its values
    
    images: [B,C,H,W]
    indices: [B,P,2] 
    """
    assert indices.size(-1) == 2, indices.size()
    if channels_last:
        images = images.permute(0, 3, 1, 2)  # [B,C,H,W]
    B, C, H, W = images.shape
    _, P, _ = indices.shape
    inds_h, inds_w = list(indices.to(torch.long).permute(2, 0, 1))  # [B,P] each
    inds_b = torch.arange(B, dtype=torch.long).unsqueeze(-1).expand(-1, P).to(inds_h)
    inds = torch.stack([inds_b, inds_h, inds_w], 0)
    values = images.permute(0, 2, 3, 1)[list(inds)]  # [B,P,C]
    return values


def soft_index(images, indices, scale_by_imsize=True):
    assert indices.shape[-1] == 2, indices.shape
    B, C, H, W = images.shape
    _, P, _ = indices.shape

    # h_inds, w_inds = indices.split([1,1], dim=-1)
    h_inds, w_inds = list(indices.permute(2, 0, 1))
    if scale_by_imsize:
        h_inds = (h_inds + 1.0) * torch.tensor(H).to(h_inds) * 0.5
        w_inds = (w_inds + 1.0) * torch.tensor(W).to(w_inds) * 0.5

    h_inds = torch.maximum(torch.minimum(h_inds, torch.tensor(H - 1).to(h_inds)), torch.tensor(0.).to(h_inds))
    w_inds = torch.maximum(torch.minimum(w_inds, torch.tensor(W - 1).to(w_inds)), torch.tensor(0.).to(w_inds))

    h_floor = torch.floor(h_inds)
    w_floor = torch.floor(w_inds)
    h_ceil = torch.ceil(h_inds)
    w_ceil = torch.ceil(w_inds)

    bot_right_weight = (h_inds - h_floor) * (w_inds - w_floor)
    bot_left_weight = (h_inds - h_floor) * (w_ceil - w_inds)
    top_right_weight = (h_ceil - h_inds) * (w_inds - w_floor)
    top_left_weight = (h_ceil - h_inds) * (w_ceil - w_inds)

    in_bounds = (bot_right_weight + bot_left_weight + top_right_weight + top_left_weight) > 0.95
    in_bounds = in_bounds.to(torch.float32)

    top_left_vals = index_into_images(images, torch.stack([h_floor, w_floor], -1))
    top_right_vals = index_into_images(images, torch.stack([h_floor, w_ceil], -1))
    bot_left_vals = index_into_images(images, torch.stack([h_ceil, w_floor], -1))
    bot_right_vals = index_into_images(images, torch.stack([h_ceil, w_ceil], -1))

    im_vals = top_left_vals * top_left_weight[..., None]
    im_vals += top_right_vals * top_right_weight[..., None]
    im_vals += bot_left_vals * bot_left_weight[..., None]
    im_vals += bot_right_vals * bot_right_weight[..., None]

    im_vals = im_vals.view(B, P, C)

    return im_vals


def compute_compatibility(positions, plateau, phenotypes=None, availability=None, noise=0.1):
    """
    Compute how well "fit" each agent is for the position it's at on the plateau,
    according to its "phenotype"

    positions: [B,P,2]
    plateau: [B,H,W,Q]
    phenotypes: [B,P,D] or None
    availability: [B,H,W,A]
    """
    B, H, W, Q = plateau.shape
    P = positions.shape[1]
    if phenotypes is None:
        raise ValueError  # TODO: check the correctness of the line below
        phenotypes = soft_index(plateau, positions)

    if availability is not None:
        assert list(availability.shape)[:-1] == list(plateau.shape)[:-1], (availability.shape, plateau.shape)
        A = availability.size(-1)
        assert P % A == 0, (P, A)
        S = P // A  # population size
        plateau = availability[..., None] * plateau[..., None, :]  # [B,H,W,A,Q]
        plateau = plateau.view(B, H, W, A * Q)

    plateau_values = soft_index(plateau.permute(0, 3, 1, 2), positions, scale_by_imsize=True)
    if noise > 0:
        plateau_values += noise * torch.rand(size=plateau_values.size(), dtype=torch.float32).to(plateau_values)

    if availability is not None:
        plateau_values = l2_normalize(plateau_values.view(B, P, A, Q))
        # inds = torch.tile(torch.eye(A)[None].expand(B,-1,-1), (1,S,1))[...,None] # [B,P,A,1]
        inds = (torch.eye(A)[None].expand(B, -1, -1)).repeat(1, S, 1)[..., None]  # [B,P,A,1]
        plateau_values = torch.sum(plateau_values * inds.to(plateau_values), dim=-2)  # [B,P,Q]
    else:
        plateau_values = l2_normalize(plateau_values)

    compatibility = torch.sum(
        l2_normalize(phenotypes) * plateau_values, dim=-1, keepdim=True)  # [B,P,1]

    return compatibility


def compute_pairwise_overlaps(masks, masks_target=None, mask_thresh=None, eps=1e-6):
    """Find overlaps between masks"""
    B, N, P = masks.shape
    if masks_target is None:
        masks_target = masks
    if mask_thresh is not None:
        masks = (masks > mask_thresh).to(torch.float32)
        masks_target = (masks_target > mask_thresh).to(torch.float32)

    ## union and intersection
    overlaps = masks[..., None] * masks_target[..., None, :]  # [B,N,P,P]
    I = overlaps.sum(dim=1)
    U = torch.maximum(masks[..., None], masks_target[..., None, :]).sum(dim=1)
    iou = I / torch.maximum(U, torch.tensor(eps, dtype=torch.float32).to(masks))  # [B,P,P]

    return iou


def compete_agents(masks, fitnesses, alive,
                   mask_thresh=0.5, compete_thresh=0.2,
                   sticky_winners=True):
    """
    Kill off agents (which mask dimensions are "alive") based on mask overlap and fitnesses of each

    args:
        masks: [B,N,P]
        fitnesses: [B,P,1]
        alive: [B,P,1]

    returns:
        still_alive: [B,P,1]
    
    """
    B, N, P = masks.shape
    assert list(alive.shape) == [B, P, 1], alive.shape
    assert list(fitnesses.shape) == [B, P, 1], fitnesses.shape

    ## find territorial disputes
    overlaps = compute_pairwise_overlaps(masks, masks_target=None, mask_thresh=mask_thresh)
    disputes = overlaps > compete_thresh  # [B,P,P] <bool>

    ## agents don't fight themselves
    disputes = torch.logical_and(
        disputes, torch.logical_not(
            torch.eye(P, dtype=torch.bool).to(disputes).unsqueeze(0).expand(B, -1, -1)))

    ## kill off the agents with lower fitness in each dispute
    killed = torch.logical_and(disputes, fitnesses < torch.transpose(fitnesses, 1, 2))

    ## once an agent wins, it always wins again
    if sticky_winners:
        winners = (alive > 0.5)
        losers = torch.logical_not(winners)

        ## winners can't lose to last round's losers
        winners_vs_losers = torch.logical_and(winners, torch.transpose(losers, 1, 2))  # [B,P,P]
        killed = torch.logical_and(killed, torch.logical_not(winners_vs_losers))

        ## losers can't overtake last round's winners
        losers_vs_winners = torch.logical_and(losers, torch.transpose(winners, 1, 2))
        losers_vs_winners_disputes = torch.logical_and(losers_vs_winners, disputes)
        killed = torch.logical_or(killed, losers_vs_winners_disputes)

    ## if an agent was killed by *any* competitor, it's dead
    killed = torch.any(killed, dim=2, keepdim=True)
    alive = torch.logical_not(killed).to(torch.float32)

    return alive


def compute_distance_weighted_vectors(vector_map, positions, mask=None, beta=1.0, eps=1e-8):
    """
    compute vectors whose values are a weighted mean of vector_map, where weights are given by distance.
    """
    B, H, W, D = vector_map.shape
    assert positions.size(-1) == 2, positions.size()
    B, P, _ = positions.shape
    N = H * W

    if mask is None:
        mask = torch.ones_like(vector_map[..., 0:1])
    else:
        assert list(mask.shape) == [B, H, W, 1]

    hw_grid = coordinate_ims(B, 0, [H, W]).view(B, N, 2)
    delta_positions = hw_grid[:, None] - positions[:, :, None]  # [B,P,N,2]
    distances = torch.sqrt(delta_positions[..., 0] ** 2 + delta_positions[..., 1] ** 2 + eps)  # [B,P,N]

    ## max distance is 2*sqrt(2)
    inv_distances = (2.0 * np.sqrt(2.0)) / (distances + eps)
    inv_distances = F.softmax(beta * inv_distances * mask.view(B, 1, N), dim=-1)  # [B,P,N]
    distance_weighted_vectors = torch.sum(
        vector_map.view(B, 1, N, D) * inv_distances[..., None], dim=2, keepdim=False)  # [B,P,D]
    return distance_weighted_vectors


def masks_from_phenotypes(plateau, phenotypes, normalize=True):
    B, H, W, Q = plateau.shape
    N = H * W
    masks = dot_product_attention(
        queries=plateau.view(B, N, Q),
        keys=phenotypes,
        normalize=normalize)
    masks = F.relu(masks)
    return masks


class Competition(nn.Module):

    def __init__(
            self,
            size=None,
            num_masks=16,
            num_competition_rounds=5,
            mask_beta=10.0,
            reduce_func=reduce_max,
            stop_gradient=True,
            stop_gradient_phenotypes=True,
            normalization_func=l2_normalize,
            sum_edges=True,
            mask_thresh=0.5,
            compete_thresh=0.2,
            sticky_winners=True,
            selection_strength=100.0,
            homing_strength=10.0,
            mask_dead_segments=True
    ):
        super().__init__()
        self.num_masks = self.M = num_masks
        self.num_competition_rounds = num_competition_rounds
        self.mask_beta = mask_beta
        self.reduce_func = reduce_func
        self.normalization_func = normalization_func

        ## stop gradients
        self.sg_func = lambda x: (x.detach() if stop_gradient else x)
        self.sg_phenotypes_func = lambda x: (x.detach() if stop_gradient_phenotypes else x)

        ## agent sampling kwargs
        self.sum_edges = sum_edges

        ## competition kwargs
        self.mask_thresh = mask_thresh
        self.compete_thresh = compete_thresh
        self.sticky_winners = sticky_winners
        self.selection_strength = selection_strength
        self.homing_strength = homing_strength
        self.mask_dead_segments = mask_dead_segments

        ## shapes
        self.B = self.T = self.BT = self.N = self.Q = None
        self.size = size  # [H,W]
        if self.size:
            assert len(self.size) == 2, self.size

    def reshape_batch_time(self, x, merge=True):

        if merge:
            self.is_temporal = True
            B, T = x.size()[0:2]
            if self.B:
                assert (B == self.B), (B, self.B)
            else:
                self.B = B

            if self.T:
                assert (T == self.T), (T, self.T)
            else:
                self.T = T

            assert B * T == (self.B * self.T), (B * T, self.B * self.T)
            if self.BT is None:
                self.BT = self.B * self.T

            return torch.reshape(x, [self.BT] + list(x.size())[2:])

        else:  # split
            BT = x.size()[0]
            assert self.B and self.T, (self.B, self.T)
            if self.BT is not None:
                assert BT == self.BT, (BT, self.BT)
            else:
                self.BT = BT

            return torch.reshape(x, [self.B, self.T] + list(x.size())[1:])

    def process_plateau_input(self, plateau):

        shape = plateau.size()
        if len(shape) == 5:
            self.is_temporal = True
            self.B, self.T, self.H, self.W, self.Q = shape
            self.N = self.H * self.W
            self.BT = self.B * self.T
            plateau = self.reshape_batch_time(plateau)
        elif (len(shape) == 4) and (self.size is None):
            self.is_temporal = False
            self.B, self.H, self.W, self.Q = shape
            self.N = self.H * self.W
            self.T = 1
            self.BT = self.B * self.T
        elif (len(shape) == 4) and (self.size is not None):
            self.is_temporal = True
            self.B, self.T, self.N, self.Q = shape
            self.BT = self.B * self.T
            self.H, self.W = self.size
            plateau = self.reshape_batch_time(plateau)
            plateau = torch.reshape(plateau, [self.BT, self.H, self.W, self.Q])
        elif len(shape) == 3:
            assert self.size is not None, \
                "You need to specify an image size to reshape the plateau of shape %s" % shape
            self.is_temporal = False
            self.B, self.N, self.Q = shape
            self.T = 1
            self.BT = self.B
            self.H, self.W = self.size
            plateau = torch.reshape(plateau, [self.BT, self.H, self.W, self.Q])
        else:
            raise ValueError("input plateau map with shape %s cannot be reshaped to [BT, H, W, Q]" % shape)

        return plateau

    def forward(self, plateau):
        """
        Find the uniform regions within the plateau map 
        by competition between visual "indices."

        args:
            plateau: [B,[T],H,W,Q] feature map with smooth "plateaus"
           
        returns:
            masks: [B, [T], H, W, M] <float> one mask in each of M channels
            agents: [B, [T], M, 2] <float> positions of agents in normalized coordinates
            alive: [B, [T], M] <float> binary vector indicating which masks are valid
            phenotypes: [B, [T], M, Q] 
            unharvested: [B, [T], H, W] <float> map of regions that weren't covered
    
        """

        ## preprocess
        plateau = self.process_plateau_input(plateau)  # [BT,H,W,Q]
        plateau = self.normalization_func(plateau)

        ## sample initial indices ("agents") from borders of the plateau map
        agents = sample_coordinates_at_borders(
            plateau.permute(0, 3, 1, 2), num_points=self.M, mask=None, sum_edges=self.sum_edges)

        ## initially all of these agents are "alive"
        alive = torch.ones_like(agents[..., -1:])  # [BT,M,1]

        ## the agents have "phenotypes" depending on where they're situated on the plateau map
        phenotypes = self.sg_phenotypes_func(
            self.normalization_func(soft_index(plateau.permute(0, 3, 1, 2), agents, scale_by_imsize=True)))

        ## the "fitness" of an agent -- how likely it is to survive competition --
        ## is how well its phenotype matches the plateau vector at its current position
        fitnesses = compute_compatibility(agents, plateau, phenotypes, availability=None, noise=0.1)

        ## compute the masks at initialization
        masks_pred = masks_from_phenotypes(plateau, phenotypes, normalize=True)

        ## find the "unharvested" regions of the plateau map not covered by agents
        unharvested = torch.minimum(self.reduce_func(masks_pred, dim=-1, keepdim=True), torch.tensor(1.0).to(masks_pred))
        unharvested = 1.0 - unharvested.view(self.BT, self.H, self.W, 1)

        for r in range(self.num_competition_rounds):
            # print("Evolution round {}".format(r + 1))

            ## compute the "availability" of the plateau map for each agent (i.e. where it can harvest from)
            alive_t = torch.transpose(alive, 1, 2)  # [BT, 1, M]
            availability = alive_t * masks_pred + (1.0 - alive_t) * unharvested.view(self.BT, self.N, 1)
            availability = availability.view(self.BT, self.H, self.W, self.M)

            ## update the fitnesses
            fitnesses = compute_compatibility(
                positions=agents,
                plateau=plateau,
                phenotypes=phenotypes,
                availability=availability)

            ## kill agents that have wandered off the map
            in_bounds = torch.all(
                torch.logical_and(agents < 1.0, agents > -1.0),
                dim=-1, keepdim=True)  # [BT,M,1]
            fitnesses *= in_bounds.to(fitnesses)

            ## break ties in fitness
            fitnesses -= 0.001 * torch.arange(self.M, dtype=torch.float32)[None, :, None].expand(self.BT, -1, -1).to(fitnesses)

            ## recompute the masks
            occupied_regions = self.sg_phenotypes_func(
                soft_index(plateau.permute(0, 3, 1, 2), agents, scale_by_imsize=True))
            masks_pred = masks_from_phenotypes(plateau, occupied_regions, normalize=True)  # [BT,N,M]

            ## have each pair of agents compete.
            ## If their masks overlap, the winner is the one with higher fitness
            alive = compete_agents(masks_pred, fitnesses, alive,
                                   mask_thresh=self.mask_thresh,
                                   compete_thresh=self.compete_thresh,
                                   sticky_winners=self.sticky_winners)

            alive *= in_bounds.to(alive)
            alive_t = torch.transpose(alive, 1, 2)

            # print("Num alive masks", alive.sum())

            ## update which parts of the plateau are "unharvested"
            unharvested = torch.minimum(self.reduce_func(masks_pred * alive_t, dim=-1, keepdim=True),
                                        torch.tensor(1.0, dtype=torch.float32).to(masks_pred))
            unharvested = 1.0 - unharvested.view(self.BT, self.H, self.W, 1)

            ## update phenotypes of the winners
            if self.mask_thresh is not None:
                winner_phenotypes = (masks_pred[..., None] > self.mask_thresh).to(plateau)
            winner_phenotypes = winner_phenotypes * plateau.view(self.BT, self.N, 1, self.Q)
            winner_phenotypes = self.normalization_func(winner_phenotypes.mean(dim=1))  # [BT,M,Q]
            phenotypes += (alive * winner_phenotypes) * self.selection_strength

            ## reinitialize losing agent positions
            alive_mask = (alive > 0.5).to(torch.float32)
            loser_agents = sample_coordinates_at_borders(
                plateau.permute(0, 3, 1, 2), num_points=self.M,
                mask=unharvested.permute(0, 3, 1, 2),
                sum_edges=self.sum_edges)
            agents = agents * alive_mask + loser_agents * (1.0 - alive_mask)

            ## reinitialize loser agent phenotypes
            loser_phenotypes = self.normalization_func(
                compute_distance_weighted_vectors(plateau, agents, mask=unharvested, beta=self.homing_strength))
            phenotypes = alive_mask * phenotypes + (1.0 - alive_mask) * loser_phenotypes
            phenotypes = self.normalization_func(phenotypes)

            ## that's it for this round!

        ## run a final competition between the surviving masks
        if self.mask_beta is not None:
            masks_pred = F.softmax(
                self.mask_beta * masks_pred * alive_t - \
                self.mask_beta * (1.0 - alive_t), dim=-1)
        if self.mask_dead_segments:
            masks_pred *= alive_t

        masks_pred = masks_pred.view(self.BT, self.H, self.W, self.M)
        if self.is_temporal:
            masks_pred = self.reshape_batch_time(plateau, merge=False)
            agents = self.reshape_batch_time(agents, merge=False)
            alive = self.reshape_batch_time(alive, merge=False)
            phenotype = self.reshape_batch_time(phenotype, merge=False)
            unharvested = self.reshape_batch_time(unharvested, merge=False)

        return (masks_pred, agents, alive, phenotypes, unharvested)


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


def segments_rgb_to_one_hot(objects, channels_last=False, max_objects=8):
    hashed_objects = object_id_hash(objects, channels_last=channels_last)
    if channels_last:
        hashed_objects = hashed_objects.permute(0, 3, 1,
                                                2)  # [B,1,H,W]
    hashed_objects = list(hashed_objects[:,
                          0])  # B-length list of [H,W]
    out = []
    for b, im in enumerate(hashed_objects):
        one_hot_im = []
        obj_ids = torch.unique(im).to(im)
        for o_id in list(obj_ids[:max_objects]):
            obj = (im == o_id)
            one_hot_im.append(obj)
        one_hot_im = torch.stack(one_hot_im, 0).to(torch.float32)
        pad = torch.zeros(size=([max_objects - int(one_hot_im.shape[0])] + list(one_hot_im.shape)[1:])).to(one_hot_im)
        one_hot_im = torch.cat([one_hot_im, pad], 0)
        print(one_hot_im.shape)
        out.append(one_hot_im)
    out = torch.stack(out, 0)
    return out


def test_on_segmentation_color(num_test_images=5):
    from PIL import Image
    from connected_component import label_connected_component
    # instantiate competition module
    competition_module = Competition(num_masks=32, num_competition_rounds=5)
    H, W = 32, 32
    # load GT images
    folder_path = '/mnt/fs6/honglinc/dataset/playroom_large_v1_main_images/objects/model_split_0'
    for i in range(num_test_images):
        image_path = os.path.join(folder_path, format(i, '04d'), format(5, '05d') + '.png')
        image = Image.open(image_path)
        image = np.asarray(image)

        image = F.interpolate(torch.tensor(image[None]).permute(0, 3, 1, 2).float().cuda(), size=[H, W], mode='nearest').long()

        plateau = segments_rgb_to_one_hot(image, channels_last=False)
        plateau = plateau.permute(0, 2, 3, 1)  # [B, H, W, Q]

        start = time.time()
        masks, agents, alive, phenotypes, unharvested = competition_module(plateau)
        cc_labels = label_connected_component(masks[0].argmax(-1))
        print('Forward time: ', time.time() - start)

        mask_inds = np.where(alive[0, :, 0].cpu())[0]
        masks = masks.cpu()
        print(np.argmax(masks[0, ...], axis=-1))
        for ind in mask_inds:
            print("num pixels in mask %d ---> %d" % (ind, (np.argmax(masks[0], -1) == ind).sum()))

        labels = np.argmax(masks[0, ...], axis=-1)

        # # --- Visualization ---
        plt.subplot(1, 3, 1)
        plt.imshow(plateau[0].argmax(-1).cpu())
        plt.title('GT: %d' % len(torch.unique(plateau[0].argmax(-1))))
        plt.subplot(1, 3, 2)
        plt.imshow(labels)
        plt.title('Output: %d' % len(torch.unique(labels)))
        plt.subplot(1, 3, 3)
        plt.imshow(cc_labels.cpu())
        plt.title('CC: %d' % len(torch.unique(cc_labels)))
        plt.show()
        plt.close()


def reorder_int_labels(x):
    _, y = torch.unique(x, return_inverse=True)
    y -= y.min()
    return y

def test_kp_comp_cc(num_test_images=4, mode='gt'):
    from PIL import Image
    from propagation import GraphPropagation, compute_gt_affinity
    from connected_component import label_connected_component
    import h5py
    # instantiate competition module
    graph_propagation = GraphPropagation()
    competition_module = Competition(num_masks=32, num_competition_rounds=3)
    H, W = 32, 32

    # load GT images
    for i in range(num_test_images):
        if mode == 'gt':
            folder_path = '/mnt/fs6/honglinc/dataset/playroom_large_v1_main_images/objects/model_split_0'
            image_path = os.path.join(folder_path, format(i, '04d'), format(5, '05d') + '.png')
            image = Image.open(image_path)
            image = np.asarray(image)

            image = F.interpolate(torch.tensor(image[None]).permute(0, 3, 1, 2).float().cuda(), size=[H, W], mode='nearest').long()
            objects, affinity = compute_gt_affinity(image, channels_last=False)
        else:
            level = 4
            exp_id = 'tdw40_p4_full_sym_P_0/eval' #'tdw_psgv2_mini_p3_full_affinity_0' #'tdw_psgv2_p4_P_0'#'tdw_psgv2_p4_P_sym'# # 'tdw_psgv2_p3_P_0'
            folder_path = '/mnt/fs6/honglinc/detectron_psg_gpu/projects/Panoptic-DeepLab/saved_tensor'
            folder_path = os.path.join(folder_path, exp_id, str(level))
            ref_points = [
                'model_split_0-0049.hdf5',
                'model_split_0-0089.hdf5',
                'model_split_0-0099.hdf5',
                'model_split_0-0029.hdf5'
            ]
            filename = ref_points[i]

            out = h5py.File(os.path.join(folder_path, filename + '.h5'), 'r')
            logits = torch.tensor(out['logits'][:]).cuda()
            image = out['image'][:].transpose(1, 2, 0)
            objects = out['labels'][:].reshape(H, W)

            affinity = logits.softmax(-1)
            affinity /= torch.max(affinity, dim=-1, keepdim=True)[0].clamp(min=1e-12)

        B, N, Q = 1, H * W, 256
        h0 = torch.randn([B, N, Q]).softmax(-1).cuda()
        start = time.time()

        plateau = graph_propagation.forward(h0, affinity)
        plateau = plateau.reshape([B, H, W, Q])
        print('KP time: ', time.time() - start)

        masks, agents, alive, phenotypes, unharvested = competition_module(plateau)
        print('Comp time: ', time.time() - start)

        cc_labels = label_connected_component(masks[0].argmax(-1))
        cc_labels = reorder_int_labels(cc_labels)

        test_pseudo_supervision(cc_labels)
        print('Total time: ', time.time() - start)

        mask_inds = np.where(alive[0, :, 0].cpu())[0]
        masks = masks.cpu()
        # print(np.argmax(masks[0, ...], axis=-1))
        # for ind in mask_inds:
        #     print("num pixels in mask %d ---> %d" % (ind, (np.argmax(masks[0], -1) == ind).sum()))

        labels = np.argmax(masks[0, ...], axis=-1)

        # # --- Visualization ---
        plt.figure(figsize=(10, 5))
        num_plots = 5
        plt.subplot(1, num_plots, 1)
        plt.imshow(image)
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1, num_plots, 2)
        plt.imshow(objects)
        plt.title('GT: %d' % len(np.unique(objects)))
        plt.axis('off')
        plt.subplot(1, num_plots, 3)
        plateau_argmax = plateau[0].argmax(-1).cpu()
        plateau_argmax = reorder_int_labels(plateau_argmax)
        plt.imshow(plateau_argmax)
        plt.axis('off')
        plt.title('KP: %d' % len(torch.unique(plateau[0].argmax(-1))))
        plt.subplot(1, num_plots, 4)
        plt.imshow(labels)
        plt.axis('off')
        plt.title('Comp: %d' % len(torch.unique(labels)))
        plt.subplot(1, num_plots, 5)
        plt.imshow(cc_labels.cpu())
        plt.axis('off')
        plt.title('CC: %d' % len(torch.unique(cc_labels)))

        plt.show()
        plt.close()

def kl_divergence(logits, labels, logits_mode='softmax', labels_mode='row_sum', reduce_sum=True):
    """
    :param logits: [B, N, K] raw logits (pre-softmax)
    :param labels: [B, N, K] binary labels
    :return: [B, N] KL divergence
    """
    B, N, K = logits.shape

    logits = logits.reshape([B * N, K])
    labels = labels.reshape([B * N, K])

    if logits_mode == 'softmax':
        logits = F.log_softmax(logits, -1)       # log probabilities
    elif logits_mode == 'log':
        logits = logits.log()
    else:
        raise ValueError

    if labels_mode == 'softmax':
        labels = torch.softmax(labels, -1)
    else:
        labels = labels / labels.sum(-1).unsqueeze(-1)   #  probabilities

    kl_div = F.kl_div(logits, labels, reduction='none')

    return kl_div.sum(-1) if reduce_sum else kl_div

def test_pseudo_supervision(inputs):
    H, W = inputs.shape
    inputs = inputs.reshape(-1)
    affinity = torch.randn([1, H * W, H * W]).cuda() * 1e-3
    affinity = torch.nn.Parameter(affinity)
    #pdb.set_trace()

    optimizer = torch.optim.Adam([affinity], lr=0.1)

    for i in range(100):
        optimizer.zero_grad()
        nonzero = inputs > 0
        cc_logits = affinity[0, nonzero][:, nonzero]
        cc_labels = inputs[nonzero] # [?]
        cc_targets = (cc_labels.unsqueeze(-1) == cc_labels.unsqueeze(-2)).float()
        cc_loss = kl_divergence(cc_logits[None], cc_targets[None])
        cc_loss = cc_loss.mean()

        cc_loss.backward()
        optimizer.step()
        print(i, cc_loss)

    pdb.set_trace()


def test_kp_comp_cc_multiple_runs(num_test_images=4, num_runs=3, mode='gt'):
    from PIL import Image
    from propagation import GraphPropagation, compute_gt_affinity
    from connected_component import label_connected_component
    from aggregate_pseudolabels import aggregate_multiple_labels
    import h5py
    # instantiate competition module
    graph_propagation = GraphPropagation()
    competition_module = Competition(num_masks=32, num_competition_rounds=3)
    H, W = 32, 32

    # load GT images
    for i in range(num_test_images):
        if mode == 'gt':
            folder_path = '/mnt/fs6/honglinc/dataset/playroom_large_v1_main_images/objects/model_split_0'
            image_path = os.path.join(folder_path, format(i, '04d'), format(5, '05d') + '.png')
            image = Image.open(image_path)
            image = np.asarray(image)

            image = F.interpolate(torch.tensor(image[None]).permute(0, 3, 1, 2).float().cuda(), size=[H, W], mode='nearest').long()
            objects, affinity = compute_gt_affinity(image, channels_last=False)
        else:
            level = 4
            exp_id = 'tdw40_p4_full_sym_P_0/eval' #'tdw_psgv2_mini_p3_full_affinity_0' #'tdw_psgv2_p4_P_0'#'tdw_psgv2_p4_P_sym'# # 'tdw_psgv2_p3_P_0'
            folder_path = '/mnt/fs6/honglinc/detectron_psg_gpu/projects/Panoptic-DeepLab/saved_tensor'
            folder_path = os.path.join(folder_path, exp_id, str(level))
            ref_points = [
                'model_split_0-0049.hdf5',
                'model_split_0-0089.hdf5',
                'model_split_0-0099.hdf5',
                'model_split_0-0029.hdf5'
            ]
            filename = ref_points[i]

            out = h5py.File(os.path.join(folder_path, filename + '.h5'), 'r')
            logits = torch.tensor(out['logits'][:]).cuda()
            image = out['image'][:].transpose(1, 2, 0)
            objects = out['labels'][:].reshape(H, W)

            affinity = logits.softmax(-1)
            affinity /= torch.max(affinity, dim=-1, keepdim=True)[0].clamp(min=1e-12)

        B, N, Q = 1, H * W, 256
        h0 = torch.randn([B, N, Q]).softmax(-1).cuda()
        start = time.time()

        pseudolabels_list = []
        for r in range(num_runs):
            plateau = graph_propagation.forward(h0, affinity, None)
            plateau = plateau.reshape([B, H, W, Q])
            print('KP time: ', time.time() - start)

            masks, agents, alive, phenotypes, unharvested = competition_module(plateau)
            print('Comp time: ', time.time() - start)

            cc_labels = label_connected_component(masks[0].argmax(-1))
            cc_labels = reorder_int_labels(cc_labels)
            pseudolabels_list.append(cc_labels.reshape(1, H * W))

        ret = aggregate_multiple_labels(pseudolabels_list)
        pdb.set_trace()
        # test_pseudo_supervision(cc_labels)
        print('Total time: ', time.time() - start)

        mask_inds = np.where(alive[0, :, 0].cpu())[0]
        masks = masks.cpu()
        # print(np.argmax(masks[0, ...], axis=-1))
        # for ind in mask_inds:
        #     print("num pixels in mask %d ---> %d" % (ind, (np.argmax(masks[0], -1) == ind).sum()))

        labels = np.argmax(masks[0, ...], axis=-1)

if __name__ == '__main__':
    # test_kp_comp_cc_gt(mode='pretrained')
    test_kp_comp_cc_multiple_runs(mode='pretrained')
    # test_on_segmentation_color()

    # --- testing on toy plateau map ----
    # Comp = Competition(num_masks=32, num_competition_rounds=5)
    # left = torch.ones(size=(32, 8)).unsqueeze(-1) * torch.tensor([1., 0.2, 0.])
    # middle = torch.ones(size=(32, 16)).unsqueeze(-1) * torch.tensor([0., 1., 0.2])
    # right = torch.ones(size=(32, 12)).unsqueeze(-1) * torch.tensor([0.1, 0., 1.])
    # plateau = torch.cat([left, middle, right], dim=-2).unsqueeze(0)
    # print(plateau.shape)
    # masks, agents, alive, phenotypes, unharvested = Comp(plateau)
    # mask_inds = np.where(alive[0, :, 0].numpy())[0]
    # print(np.argmax(masks[0, ...], axis=-1))
    # for ind in mask_inds:
    #     print("num pixels in mask %d ---> %d" % (ind, (np.argmax(masks[0], -1) == ind).sum()))
