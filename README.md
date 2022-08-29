# EISEN

This repository contains the official implementation for the paper "Unsupervised Segmentation in Real-World Images via Spelke Object Inference" (ECCV 2022 Oral). 

<figure style="max-width:842px;" class="image-figure">
<img src="./teaser.gif" alt="teaser image"/>
</figure>

Paper: [https://arxiv.org/abs/2205.08515](https://arxiv.org/abs/2205.08515) \
Project website: [https://neuroailab.github.io/eisen/](https://neuroailab.github.io/eisen/) \
Authors: Honglin Chen, Rahul Venkatesh, Yoni Friedman, Jiajun Wu, Joshua B. Tenenbaum, Daniel L. K. Yamins, Daniel M. Bear from Stanford University and MIT.

## Environment
Dependency:
- Pytorch 1.11
- CUDA 11.3
- [detectron2](https://github.com/facebookresearch/detectron2)
- [cc_torch](https://github.com/zsef123/Connected_components_PyTorch)

We recommend installation using Conda. Please make sure you have NVIDIA drivers supporting CUDA 11.3, or modify the version in `environment.yml. 

```
conda env create -f environment.yml
conda activate eisen
pip install 'git+https://github.com/zsef123/Connected_components_PyTorch.git'
pip install 'git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'
```

## Dataset
To train the model on the ThreeWorld Playroom dataset (30G), please download the dataset and unzip it using the following command:
```
mkdir datasets
cd datasets
wget https://www.dropbox.com/s/yqw71eir9sjq4k4/Playroom.zip
unzip Playroom.zip
```

## Precompute optical flow (recommended)
You can run the following command to precompute optical flow, which gives approximately 2x speed-up in training time. 
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset playroom --batch_size 8 --precompute_flow
```
Alternative, precomputed optical flow (71G) can be downloaded directly:
```
cd datasets/Playroom
wget https://www.dropbox.com/s/5k9x5pahusdd0zk/Playroom_flows.zip
unzip Playroom_flows.zip
```

## Training
The training script can be run using the command below. By default, it assumes optical flows have been precomputed. Otherwise, you can simply add the flag `--compute-flow` to compute optical flows during training.
```
python main.py --name eisen --dataset playroom --batch_size 8
```

## Evaluation
You can evaluate a training model using the command below. The checkpoints will be stored under `./checkpoints/`.
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset playroom --batch_size 1 --eval_only --ckpt /path/to/checkpoint
```

## Pretrained models
|  Dataset |      Backbone      | Input <br/>resolution | Output <br/>resolution |  mIoU  |                                    Model weights                                     |
|:--------------------:|:------------------:|:---------------------:|:----------------------:|:------:|:------------------------------------------------------------------------------------:|
| ThreeDWorld Playroom |  ResNet50-DeepLab  |        512x512        |        128x128         | 0.729  |  [weights](https://www.dropbox.com/s/dh3er2qfo2euswp/tdw_playroom_128x128_ckpt.pth)  |

## TODO
- [ ] Support for bootstrapped training (coming soon)
- [ ] Support for training on Bridge dataset (coming soon)

## Bibtex
```
@InProceedings{chen2022unsupervised,
author = {Chen, Honglin and Venkatesh, Rahul and Friedman, Yoni and Wu, Jiajun and Tenenbaum, Joshua B and Yamins, Daniel LK and Bear, Daniel M},
title = {Unsupervised Segmentation in Real-World Images via Spelke Object Inference},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
year = {2022}
}
```

## Acknowledgement
Our framework uses [RAFT](https://github.com/princeton-vl/RAFT) for computing optical flow from videos. Code related to connected component algorithm is adapted from [Connected_components_Pytorch](https://github.com/zsef123/Connected_components_PyTorch). Some code related to synchronized batch normalization is adpated from [sync_batchnorm](https:/github.com/zengxianyu/sync_batchnorm).
