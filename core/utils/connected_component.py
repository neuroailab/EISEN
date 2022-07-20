from cc_torch import connected_components_labeling
import torch
from core.utils.utils import reorder_int_labels

def reorder_int_labels(x):
    _, y = torch.unique(x, return_inverse=True)
    y -= y.min()
    return y

def label_connected_component(labels, min_area=20, topk=256):
    size = labels.size()
    assert len(size) == 2
    max_area = size[0] * size[1] - 1

    # per-label binary mask
    unique_labels = torch.unique(labels).reshape(-1, 1, 1)  # [?, 1, 1], where ? is the number of unique id
    binary_masks = (labels.unsqueeze(0) == unique_labels).float()  # [?, H, W]

    # label connected components
    # cc is an integer tensor, each unique id represents a single connected component
    cc = torch.cat([connected_components_labeling(binary_masks[i].byte())[None] for i in range(binary_masks.shape[0])]).unsqueeze(1)

    # reorder indices in cc so that cc_area tensor below is a smaller
    cc = reorder_int_labels(cc)

    # area of each connected components
    cc_area = torch.bincount(cc.long().flatten().cpu()).cuda()  # bincount on GPU is much slower
    num_cc = cc_area.shape[0]
    valid = (cc_area >= min_area) & (cc_area <= max_area)  # [num_cc]

    if num_cc < topk:
        selected_cc = torch.arange(num_cc).cuda()
    else:
        _, selected_cc = torch.topk(cc_area, k=topk)
        valid = valid[selected_cc]


    # collapse the 0th dimension, since there is only matched one connected component (across 0th dimension)
    cc_mask = (cc == selected_cc.reshape(1, -1, 1, 1)).sum(0)  # [num_cc, H, W]
    cc_mask = cc_mask * valid.reshape(-1, 1, 1)
    out = cc_mask.argmax(0)

    out = reorder_int_labels(out)  # [H, W]
    return out

def filter_small_connected_component(labels, min_area=10, invalid_value=0):
    size = labels.size()
    assert len(size) == 2
    # max_area = size[0] * size[1] - 1

    # per-label binary mask
    unique_labels = torch.unique(labels).reshape(-1, 1, 1)  # [?, 1, 1], where ? is the number of unique id
    binary_masks = (labels.unsqueeze(0) == unique_labels).float()  # [?, H, W]

    # filter the binary mask first
    # if the binary mask has area smaller than min_area, then its CC must be smaller than min_area
    area = binary_masks.flatten(1, 2).sum(-1)
    valid_area_mask = area > min_area

    if valid_area_mask.sum() < valid_area_mask.shape[0]:  # filter
        invalid_label_mask = binary_masks[~valid_area_mask].sum(0) > 0
        labels = invalid_label_mask * invalid_value + ~invalid_label_mask * labels
        binary_masks = binary_masks[valid_area_mask]


    # label connected components
    # cc is an integer tensor, each unique id represents a single connected component
    cc = connected_components(binary_masks.unsqueeze(1), num_iterations=200)  # [?, 1, H, W]

    # reorder indices in cc so that cc_area tensor below is a smaller
    cc = reorder_int_labels(cc)

    # area of each connected components
    cc_area = torch.bincount(cc.long().flatten().cpu()).cuda()  # bincount on GPU is much slower
    num_cc = cc_area.shape[0]
    cc_idx = torch.arange(num_cc).cuda()
    assert torch.equal(cc.sum(0), cc.max(0)[0])  # make sure the CCs are mutually exclusive
    cc = cc.sum(0)  # collapse the 0th dimension (note: the CCs must be mutually exclusive)

    # find ccs that are greater than min area and set the invalid ones to zeros
    # there are two ways to implement it -- choose the one that is more memory efficient
    num_valid_segments = (cc_area >= min_area).sum()
    num_invalid_segments = (cc_area < min_area).sum()

    if num_valid_segments < num_invalid_segments:
        valid = cc_area >= min_area
        valid_cc_idx = cc_idx[valid].view(-1, 1, 1)
        valid_mask = (cc == valid_cc_idx).sum(0) > 0
        valid_labels = valid_mask * labels + ~valid_mask * invalid_value
    else:
        invalid = cc_area < min_area
        invalid_cc_idx = cc_idx[invalid].view(-1, 1, 1)
        invalid_mask = (cc == invalid_cc_idx).sum(0) > 0
        valid_labels = ~invalid_mask * labels + invalid_mask * invalid_value

    return valid_labels


