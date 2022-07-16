import numpy as np
import scipy
import sklearn.metrics
import skimage
from skimage.segmentation.boundaries import find_boundaries
from sklearn.cluster import KMeans

import torch
from torchvision import transforms
import torch.nn.functional as F
import pdb

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
        objects = objects.permute(0,2,3,1)
    C = objects.shape[-1]
    val = torch.tensor(val, dtype=dtype_out)
    objects = objects.to(dtype_out)
    out = torch.zeros_like(objects[...,0:1])
    for c in range(C):
        scale = torch.pow(val, C-1-c)
        out += scale * objects[...,c:c+1]
    if not channels_last:
        out = out.permute(0,3,1,2)

    return out

class SegmentationMetrics(object):
    """
    A class for computing metrics given a pair of pred and gt segment maps
    """
    def __init__(
            self,
            gt_objects, # the true segmentation
            pred_objects=None, # the predicted segmentation
            background_value=0, # value of background segment
            min_gt_size=1, # num pixels needed to be a true segment
            size=None, # image size to do evaluation at
            max_objects=None,
            exclude_pred_ids=None,
    ):
        ## attributes for all evaluations
        self.size = size
        self.background_value = background_value
        self.min_gt_size = min_gt_size
        self.max_objects = max_objects

        ## set attributes of the gt and resize
        self.gt_objects = gt_objects
        self.pred_objects = pred_objects

        ## initialize metrics
        self.best_ious = None
        self.mean_ious = None
        self.recalls = None
        self.boundary_f1_scores = None
        self.mean_boundary_f1_scores = None
        self.exclude_pred_ids = exclude_pred_ids

    @property
    def gt_objects(self):
        return self._gt_objects
    @gt_objects.setter
    def gt_objects(self, value):
        self._set_gt_objects(value)
        self._set_gt_ids()

    @property
    def pred_objects(self):
        return self._pred_objects
    @pred_objects.setter
    def pred_objects(self, value):
        self._set_pred_objects(value)

    def _object_id_hash(self, objects, dtype_out=np.int32, val=256):
        C = objects.shape[-1]
        out = np.zeros(shape=objects.shape[:-1], dtype=dtype_out)
        for c in range(C):
            scale = np.power(val, C-1-c)
            out += scale * objects[...,c]
        return out

    def _parse_objects_tensor(self, objects):

        shape = list(objects.shape)
        if len(shape) == 2:
            objects = objects[...,None]

        dtype = objects.dtype
        if dtype == torch.uint8:
            assert (shape[-1] == 3) or (shape[-3] == 3), shape
            channels_last = True if shape[-1] == 3 else False
        else:
            assert dtype == torch.int32, dtype
            if (shape[-1] == 1) or (shape[-3] == 1):
                channels_last = True if shape[-1] == 1 else False
            else: # 3 channels
                objects = objects[...,None]
                channels_last = True
                shape = shape + [1]

        self._temporal = False
        if len(shape) == 3:
            objects = objects[None]
            self.B = 1
            self.T = 1
            self.BT = self.B
        elif len(shape) == 5:
            self._temporal = True
            self.B, self.T = shape[:2]
            self.BT = self.B*self.T
            objects = objects.view(self.BT,*shape[2:])
        else:
            assert len(objects.shape) == 4, "torch objects must have shape [BT,C,H,W] or [BT,H,W,C]"
            self.B = shape[0]
            self.T = 1
            self.BT = self.B

        if self.max_objects is None:
            if dtype == torch.uint8:
                hashed = object_id_hash(objects, channels_last=channels_last)
            else:
                hashed = objects
            ims = list(hashed)
            num_objects = [int(torch.unique(im).size(0)) for im in ims]
            self.max_objects = max(num_objects)

        if dtype == torch.uint8:
            objects = object_id_hash(objects, channels_last=channels_last)

        if not channels_last:
            objects = objects.permute(0,2,3,1)

        if self.size is not None:
            objects = F.interpolate(objects.permute(0,3,1,2).float(), size=self.size, mode='nearest').permute(0,2,3,1).int()

        assert objects.dtype == torch.int32, objects.dtype
        return objects.numpy()

    def _parse_objects_array(self, objects):
        if objects.shape[-1] not in [1,3]:
            objects = objects[...,None]
        if objects.shape[-1] == 3:
            assert objects.dtype == np.uint8, objects.dtype
            objects = self._object_id_hash(objects)
        else:
            assert objects.dtype == np.int32

        self._temporal = False
        if len(objects.shape) == 5:
            self._temporal = True
            self.B,self.T = objects.shape[:2]
            self.BT = self.B*self.T
            objects = objects.reshape([self.BT] + objects.shape[2:])
        elif len(objects.shape) == 3:
            self.B = objects.shape[0]
            self.T = 1
            self.BT = self.B
            objects = objects[...,None]
        else:
            assert len(objects.shape) == 4, objects.shape
            self.B = objects.shape[0]
            self.T = 1
            self.BT = self.B

        if self.size is not None:
            objects = map(lambda im: skimage.transform.resize(im.astype(float), self.size, order=0).astype(np.int32), [objects[ex] for ex in range(self.BT)])
            objects = np.stack(objects, 0)

    def _set_gt_objects(self, objects):
        if isinstance(objects, torch.Tensor):
            objects = self._parse_objects_tensor(objects)
        else:
            objects = self._parse_objects_array(objects)

        assert len(objects.shape) == 4, objects.shape
        assert objects.shape[-1] == 1, objects.shape
        assert objects.dtype == np.int32, objects.dtype

        self._gt_objects = objects[...,0]
        self.gt_shape = self._gt_objects.shape
        self.size = self.gt_shape[-2:]

    def _set_gt_ids(self):
        self.gt_ids = []
        for ex in range(self.BT):
            self.gt_ids.append(
                np.unique(self.gt_objects[ex]))


    def _set_pred_objects(self, objects):
        if objects is None:
            return
        if isinstance(objects, torch.Tensor):
            objects = self._parse_objects_tensor(objects)
        else:
            objects = self._parse_objects_array(objects)

        assert len(objects.shape) == 4, objects.shape
        assert objects.shape[-1] == 1, objects.shape
        assert objects.dtype == np.int32, objects.dtype

        ## subtract off the minimum value
        offsets = objects.min(axis=(1,2), keepdims=True)
        objects -= offsets

        self._pred_objects = objects[...,0]


    def _get_mask(self, objects, obj_id=0):
        return objects == obj_id

    def get_gt_mask(self, ex, t=0, obj_id=0):
        b = ex*self.T + t
        return self._get_mask(self.gt_objects[b], obj_id)

    def get_pred_mask(self, ex, t=0, obj_id=0):
        assert self.pred_objects is not None
        b = ex*self.T + t
        return self._get_mask(self.pred_objects[b], obj_id)

    def get_background(self, ex, t=0):
        return self.get_gt_mask(ex, t, self.background_value)

    @staticmethod
    def mask_IoU(pred_mask, gt_mask, min_gt_size=1):
        """Compute intersection over union of two boolean masks"""
        assert pred_mask.shape == gt_mask.shape, (pred_mask.shape, gt_mask.shape)
        assert pred_mask.dtype == gt_mask.dtype == bool, (pred_mask.dtype, gt_mask.dtype)
        num_gt_px = gt_mask.sum()
        num_pred_px = pred_mask.sum()
        if num_gt_px < min_gt_size:
            return np.nan

        overlap = (pred_mask & gt_mask).sum().astype(float)
        IoU = overlap / (num_gt_px + num_pred_px - overlap)
        return IoU

    @staticmethod
    def mask_precision(pred_mask, gt_mask, min_gt_size=1):
        assert pred_mask.shape == gt_mask.shape, (pred_mask.shape, gt_mask.shape)
        assert pred_mask.dtype == gt_mask.dtype == bool, (pred_mask.dtype, gt_mask.dtype)
        num_gt_px = gt_mask.sum()
        num_pred_px = pred_mask.sum()
        if num_gt_px < min_gt_size:
            return np.nan

        overlap = (pred_mask & gt_mask).sum().astype(float)
        precision = overlap / np.maximum(num_pred_px, 1.0)
        return precision

    @staticmethod
    def mask_recall(pred_mask, gt_mask, min_gt_size=1):
        assert pred_mask.shape == gt_mask.shape, (pred_mask.shape, gt_mask.shape)
        assert pred_mask.dtype == gt_mask.dtype == bool, (pred_mask.dtype, gt_mask.dtype)
        num_gt_px = gt_mask.sum()
        num_pred_px = pred_mask.sum()
        if num_gt_px < min_gt_size:
            return np.nan

        overlap = (pred_mask & gt_mask).sum().astype(float)
        recall = overlap / np.maximum(num_gt_px, 1.0)
        return recall

    def _mask_metrics(self):
        return {'iou': self.mask_IoU, 'precision': self.mask_precision, 'recall': self.mask_recall}

    def compute_matched_IoUs(self, pred_objects=None, exclude_gt_ids=[], metric='iou'):
        if pred_objects is not None:
            self.pred_objects = pred_objects

        assert metric in ['iou', 'precision', 'recall'], "Metric must be 'iou', 'precision', or 'recall'"
        metric_func = self._mask_metrics()[metric]

        exclude_ids = list(set(exclude_gt_ids + [self.background_value]))
        best_IoUs = []
        best_pred_objs = []
        matched_preds, matched_gts = [], []

        for b in range(self.BT):

            ex, t = (b // self.T, b % self.T)

            # the ids in each gt mask
            ids_here = [o for o in self.gt_ids[b] if o not in exclude_ids]
            num_gt = len(ids_here)

            # the pred masks
            if self.exclude_pred_ids is None:
                preds = map(lambda o_id: self.get_pred_mask(ex, t, o_id),
                            sorted(list(np.unique(self.pred_objects[b]))))
            else:
                preds = map(lambda o_id: self.get_pred_mask(ex, t, o_id),
                            sorted([i for i in list(np.unique(self.pred_objects[b])) if i not in self.exclude_pred_ids]))
            preds = list(preds)

            num_preds = len(preds)

            # # ---- visualize ----
            # import pdb;pdb.set_trace()
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(20, 5))
            # for i in range(num_preds):
            #     plt.subplot(1, num_preds, i+1)
            #     plt.imshow(preds[i])
            #     plt.title('Pred %d' % i)
            # plt.show()
            # plt.close()
            # plt.figure(figsize=(20, 5))
            # for i in range(num_gt):
            #     plt.subplot(1, num_gt, i+1)
            #     plt.imshow(self.get_gt_mask(ex, t, ids_here[i]))
            #     plt.title('GT %d' % i)
            # plt.show()
            # plt.close()

            # compute full matrix of ious
            gts = []
            ious = np.zeros((num_gt, num_preds), dtype=np.float32)
            for m in range(num_gt):
                gt_mask = self.get_gt_mask(ex, t, ids_here[m])
                gts.append(gt_mask)
                for n in range(num_preds):
                    pred_mask = preds[n]
                    # pdb.set_trace()
                    iou = metric_func(pred_mask, gt_mask, self.min_gt_size)
                    ious[m,n] = iou if not np.isnan(iou) else 0.0

            # linear assignment
            gt_inds, pred_inds = scipy.optimize.linear_sum_assignment(1.0 - ious)

            # output values
            best = np.array([0.0] * len(ids_here))
            best[gt_inds] = ious[gt_inds, pred_inds]
            best_IoUs.append(best)
            best_objs = np.array([0] * len(ids_here))
            best_objs[gt_inds] = np.array([sorted(list(np.unique(self.pred_objects[b])))[i] for i in pred_inds])
            best_pred_objs.append(best_objs)

            count = 0
            matched_pred = []
            for m in range(num_gt):
                if count < len(gt_inds):
                    if m == gt_inds[count]:
                        matched_pred.append(preds[pred_inds[count]])
                        count += 1
                        continue

                matched_pred.append(np.zeros_like(preds[0]))
            matched_preds.append(matched_pred)
            # # # # ---- visualize ----
            # import pdb;pdb.set_trace()
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(20, 5))
            # for i in range(len(matched_preds[0])):
            #     plt.subplot(1, len(matched_preds[0]), i+1)
            #     plt.imshow(matched_preds[0][i])
            #     plt.title('Pred %d' % i)
            # plt.show()
            # plt.close()
            # plt.figure(figsize=(20, 5))
            # for i in range(len(matched_preds[0])):
            #     plt.subplot(1, len(matched_preds[0]), i+1)
            #     plt.imshow(matched_gts[0][i])
            #     plt.title('GT %d' % i)
            # plt.show()
            # plt.close()
            # print(best_IoUs, best[gt_inds])
            #
            # pdb.set_trace()

        self.best_ious = best_IoUs
        self.best_object_ids = best_pred_objs
        self.seg_out = (matched_preds, [gts], best_IoUs)

        return self.mean_ious

    def compute_best_IoUs(self, pred_objects=None):
        raise NotImplementedError("Compute the best possible IoUs, reusing pred objects if needed")

    def compute_recalls(self, pred_objects=None, thresh=0.5, exclude_gt_ids=[]):
        if pred_objects is not None:
            self.pred_objects = pred_objects

        if self.best_ious is None:
            self.compute_best_IoUs(exclude_gt_ids=exclude_gt_ids)

        recalls = np.zeros((self.BT), dtype=np.float32)
        for b in range(self.BT):
            true_pos = np.array(self.best_ious[b]) >= thresh
            recall = (true_pos.sum().astype(float) / len(true_pos)) if len(true_pos) else np.nan
            recalls[b] = recall

        self.recalls = recalls
        self.mean_recalls = np.nanmean(self.recalls)
        return self.mean_recalls

    def compute_boundary_f_measures(self, pred_objects=None, stride=1, connectivity=2, mode='thick',
                                    exclude_gt_ids=[]):

        """
        For matched pred and gt masks, compute F measure on their boundary pixels.
        F measure is defined as 2*(precision * recall) / (precision + recall)
        """
        if pred_objects is not None:
            self.pred_objects = pred_objects

        if self.best_object_ids is None:
            self.compute_matched_IoUs()

        exclude_ids = exclude_gt_ids + [self.background_value]

        boundary_fs = []
        for b in range(self.BT):
            ex, t = (b // self.T, b % self.T)

            ## the ground truth
            gt_ids_here = [o for o in self.gt_ids[b] if o not in exclude_ids]
            num_gt = len(gt_ids_here)

            ## get the object ids that best matched gt
            matched_objs = self.best_object_ids[b]
            num_pred = len(matched_objs)

            boundary_f = []
            for i,o_id in enumerate(matched_objs):
                gt_mask = self.get_gt_mask(ex, t, gt_ids_here[i])
                pred_mask = self.get_pred_mask(ex, t, o_id)

                gt_boundary = find_boundaries(gt_mask, connectivity=connectivity, mode=mode)
                pred_boundary = find_boundaries(pred_mask, connectivity=connectivity, mode=mode)

                ## precision and recall and F1
                true_pos = (gt_boundary & pred_boundary).sum().astype(float)
                false_pos = (~gt_boundary & pred_boundary).sum().astype(float)
                false_neg = (gt_boundary & (~pred_boundary)).sum().astype(float)
                precision = true_pos / (true_pos + false_pos) if (true_pos > 0.0) else 1.0 - (false_pos > 0.0).astype(float)
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg > 0.0) else 1.0
                F1 = (2 * precision * recall) / (precision + recall) if (precision + recall > 0.0) else 0.0
                boundary_f.append(F1)

            ## if there were fewer pred objects than gt
            if num_pred < num_gt:
                boundary_f.extend([0.0] * (num_gt - num_pred))

            boundary_fs.append(np.array(boundary_f))

        self.boundary_f1_scores = boundary_fs

        return self.mean_boundary_f1_scores

    @property
    def mean_ious(self):
        if self.best_ious is None:
            return None
        elif self._mean_ious is None:
            self._mean_ious = np.array([np.nanmean(self.best_ious[b]) for b in range(self.BT)])
            if self._temporal:
                self._mean_ious = self._mean_ious.reshape((self.B, self.T))
            return self._mean_ious
        else:
            return self._mean_ious
    @mean_ious.setter
    def mean_ious(self, value=None):
        if value is not None:
            raise ValueError("You can't set the mean ious, you need to compute it")
        self._mean_ious = value

    @property
    def mean_boundary_f1_scores(self):
        if self.boundary_f1_scores is None:
            return None
        elif self._mean_boundary_f1_scores is None:
            self._mean_boundary_f1_scores = np.array(
                [np.nanmean(self.boundary_f1_scores[b]) for b in range(self.BT)])
            if self._temporal:
                self._mean_boundary_f1_scores = self._mean_boundary_f1_scores.reshape((self.B, self.T))
            return self._mean_boundary_f1_scores
        else:
            return self._mean_boundary_f1_scores

    @mean_boundary_f1_scores.setter
    def mean_boundary_f1_scores(self, value=None):
        if value is not None:
            raise ValueError("You need to compute boundary_f_measure")
        self._mean_boundary_f1_scores = value


def measure_static_segmentation_metric(out, inputs, size, segment_key,
                                       eval_full_res=False, moving_only=True, exclude_zone=True,
                                       exclude_pred_ids=None, gt_seg=None):

    if gt_seg is not None:
        gt_objects = gt_seg.int()
    else:
        gt_objects = inputs['gt_segment'].int()
    assert gt_objects.max() < torch.iinfo(torch.int32).max, gt_objects
    if not eval_full_res:
        gt_objects = F.interpolate(gt_objects.float().unsqueeze(1), size=size, mode='nearest').int()

    exclude_values = []
    if not isinstance(segment_key, list):
        segment_key = [segment_key]

    segment_metric = {}
    segment_out = {}
    for key in segment_key:
        results = {'mean_ious': [], 'recalls': [], 'boundary_f1_scores': []}
        pred_objects = out[key]
        pred_objects = pred_objects.reshape(pred_objects.shape[0], 1, size[0], size[1])

        metric = SegmentationMetrics(gt_objects=gt_objects.cpu(),
                                     pred_objects=pred_objects.int().cpu(),
                                     size=None if eval_full_res else size,
                                     background_value=0,
                                     exclude_pred_ids=exclude_pred_ids)

        metric.compute_matched_IoUs(exclude_gt_ids=list(set([0] + exclude_values)))
        metric.compute_recalls()
        metric.compute_boundary_f_measures(exclude_gt_ids=list(set([0] + exclude_values)))

        results['mean_ious'].append(metric.mean_ious)
        results['recalls'].append(metric.recalls)
        results['boundary_f1_scores'].append(metric.mean_boundary_f1_scores)

        for k, v in results.items():
            segment_metric[f'metric_{key}_{k}'] = torch.tensor(np.mean(v))
        segment_out[key] = metric.seg_out

    return segment_metric, segment_out



def four_quadrant_segments(size=[128,128], separator=[0.5, 0.5], minval=1, maxval=32):
    H,W = size
    h1 = int(H * separator[0])
    h2 = H-h1
    w1 = int(W * separator[1])
    w2 = W-w1

    vals = torch.randint(size=[4], low=minval, high=maxval, dtype=torch.int32)
    q1 = torch.ones([h1,w1]).to(vals) * vals[0]
    q2 = torch.ones([h1,w2]).to(vals) * vals[1]
    q3 = torch.ones([h2,w2]).to(vals) * vals[2]
    q4 = torch.ones([h2,w1]).to(vals) * vals[3]
    top = torch.cat([q2, q1], dim=1)
    bottom = torch.cat([q3,q4], dim=1)
    out = torch.cat([top,bottom], dim=0)[None]
    return out

if __name__ == '__main__':
    size = [128,128]
    # gt_objects = torch.randint(size=(4,2,3,32,32), low=0, high=255, dtype=torch.uint8)
    # gt_objects = torch.randint(size=(4,2,1,32,32), low=0, high=8, dtype=torch.int32)
    # pred_objects = torch.randint(size=(4,2,16,16), low=0, high=32, dtype=torch.int32)
    gt_objects = four_quadrant_segments(size, separator=[0.3,0.6])
    Metrics = SegmentationMetrics(gt_objects, pred_objects=gt_objects, size=size)
    print("gt", Metrics.gt_objects.shape, Metrics.gt_objects.dtype)
    print("pred", Metrics.pred_objects.shape, Metrics.pred_objects.dtype)
    print("B, T, size", Metrics.B, Metrics.T, Metrics.size)

    Metrics.compute_matched_IoUs()
    print("Best ious", Metrics.best_ious)
    print("Best objects", Metrics.best_object_ids)

    Metrics.compute_recalls()
    print("recall", Metrics.recalls)

    # Metrics.compute_matched_IoUs(gt_objects)
    # print("Best ious", Metrics.best_ious)
    # print("Best objects", Metrics.best_object_ids)

    # Metrics.compute_recalls()
    # print("recall", Metrics.recalls)

    print("mean ious", Metrics.mean_ious)
    Metrics.compute_boundary_f_measures()
    print("boundary f1", Metrics.boundary_f1_scores)
    print("boundary f1", Metrics.mean_boundary_f1_scores)

    # print("mask precision", Metrics.mask_precision(Metrics.pred_objects[0] == Metrics.gt_ids[0][0], Metrics.gt_objects[0] == Metrics.gt_ids[0][0]))
    # print("mask recall", Metrics.mask_recall(Metrics.pred_objects[0] == Metrics.gt_ids[0][0], Metrics.gt_objects[0] == Metrics.gt_ids[0][0]))

    Metrics.compute_matched_IoUs(metric='precision')
    print("Mask precision", Metrics.best_ious)

    Metrics.compute_matched_IoUs(metric='recall')
    print("Mask recall", Metrics.best_ious)    
    
    