import os
import json
import glob
import pdb
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import shutil
import logging

class PlayroomDataset(Dataset):
    def __init__(self, training, args, frame_idx=5, dataset_dir = './datasets/Playroom'):

        self.training = training
        self.frame_idx = frame_idx
        self.args = args

        # meta.json is only required for TDW datasets
        meta_path = os.path.join(dataset_dir, 'meta.json')
        self.meta = json.loads(Path(meta_path).open().read())

        if self.training:
            self.file_list = glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-9]*', '*[0-8]'))
        else:
            self.file_list = glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-3]', '*9'))

        if args.precompute_flow: # precompute flows for training and validation dataset
            self.file_list = glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-3]', '*9')) # glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-9]*', '*[0-8]')) #+ \


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        frame_idx = self.frame_idx if os.path.exists(self.get_image_path(file_name, self.frame_idx)) else 0
        img1 = read_image(self.get_image_path(file_name, frame_idx))

        flag = os.path.exists(self.get_image_path(file_name, self.frame_idx+1))
        img2 = read_image(self.get_image_path(file_name, frame_idx+1)) if flag else img1
        segment_colors = read_image(self.get_image_path(file_name.replace('/images/', '/objects/'), frame_idx))
        gt_segment = self.process_segmentation_color(segment_colors, file_name)

        ret = {'img1': img1, 'img2': img2, 'gt_segment': gt_segment}

        if not self.args.compute_flow and not self.args.precompute_flow:
            flow = np.load(os.path.join(file_name, 'flow_'+format(frame_idx, '05d') + '.npy'))
            magnitude = torch.tensor((flow ** 2).sum(0, keepdims=True) ** 0.5)
            segment_target = (magnitude > self.args.flow_threshold)
            ret['segment_target'] = segment_target
        elif self.args.precompute_flow:
            ret['file_name'] = self.get_image_path(file_name, frame_idx)

        return ret

    @staticmethod
    def get_image_path(file_name, frame_idx):
        return os.path.join(file_name, format(frame_idx, '05d') + '.png')

    @staticmethod
    def _object_id_hash(objects, val=256, dtype=torch.long):
        C = objects.shape[0]
        objects = objects.to(dtype)
        out = torch.zeros_like(objects[0:1, ...])
        for c in range(C):
            scale = val ** (C - 1 - c)
            out += scale * objects[c:c + 1, ...]
        return out

    def process_segmentation_color(self, seg_color, file_name):
        # convert segmentation color to integer segment id
        raw_segment_map = self._object_id_hash(seg_color, val=256, dtype=torch.long)
        raw_segment_map = raw_segment_map.squeeze(0)

        # remove zone id from the raw_segment_map
        meta_key = 'playroom_large_v3_images/' + file_name.split('/images/')[-1] + '.hdf5'
        zone_id = int(self.meta[meta_key]['zone'])
        raw_segment_map[raw_segment_map == zone_id] = 0

        # convert raw segment ids to a range in [0, n]
        _, segment_map = torch.unique(raw_segment_map, return_inverse=True)
        segment_map -= segment_map.min()

        return segment_map


def fetch_dataloader(args, training=True, drop_last=True):
    """ Create the data loader for the corresponding trainign set """
    if args.dataset == 'playroom':
        dataset = PlayroomDataset(training=training, args=args)
    else:
        raise ValueError(f'Expect dataset in [playroom], but got {args.dataset} instead')

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            pin_memory=False,
                            shuffle=training,
                            num_workers=args.num_workers,
                            drop_last=drop_last)

    logging.info(f'Load dataset [{args.dataset}] with {len(dataset)} image pairs')
    return dataloader


if __name__ == "__main__":
    pass