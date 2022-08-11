# EISEN

This repository contains the official implementation for the paper "Unsupervised Segmentation in Real-World Images via Spelke Object Inference" (ECCV 2022 Oral). 

<figure style="max-width:842px;" class="image-figure">
<img src="./teaser.gif" alt="teaser image"/>
</figure>

Paper: [https://arxiv.org/abs/2205.08515](https://arxiv.org/abs/2205.08515) \
Project website: [https://neuroailab.github.io/eisen/](https://neuroailab.github.io/eisen/) \
Authors: Honglin Chen, Rahul Venkatesh, Yoni Friedman, Jiajun Wu, Joshua B. Tenenbaum, Daniel L. K. Yamins, Daniel M. Bear from Stanford University and MIT.

## Environment
We have tested our code using Pytorch 1.11 and CUDA 11.3. We recommend installation using Conda. Please make sure you have NVIDIA drivers supporting CUDA 11.3, or modify the version in `environment.yml. 

```
conda env create -f environment.yml
conda activate eisen
```

## Dataset
To train the model on the ThreeWorld Playroom dataset (28G), please download the dataset and unzip it using the following command:
```
cd datasets
wget https://www.dropbox.com/s/wsogq49q7r53vhw/Playroom.zip
unzip Playroom.zip
```

## Training
The training script can be run using the following command. The training takes about 1 day on 4 RTX 3090 GPUs. 
```
python main.py --dataset playroom --batch_size 8
```

## Evaluation
You can evaluate a training model using the following command: 
```
python main.py --dataset playroom --barch_size 1 --eval_only --ckpt /path/to/checkpoint
```

## Bibtex
```
@article{chen2022unsupervised,
  title={Unsupervised Segmentation in Real-World Images via Spelke Object Inference},
  author={Chen, Honglin and Venkatesh, Rahul and Friedman, Yoni and Wu, Jiajun and Tenenbaum, Joshua B
  and Yamins, Daniel LK and Bear, Daniel M},
  journal={arXiv preprint arXiv:2205.08515},
  year={2022}
}
```

## Acknowledgement
Our framework uses [RAFT](https://github.com/princeton-vl/RAFT) for computing optical flow from videos. Code related to connected component algorithm is adapted from [Connected_components_Pytorch](https://github.com/zsef123/Connected_components_PyTorch). Some code related to synchronized batch normalization is adpated from [sync_batchnorm](https:/github.com/zengxianyu/sync_batchnorm).
