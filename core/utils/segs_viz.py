import numpy as np
import torch
import matplotlib.pyplot as plt
import cmocean.cm as cmo
from tqdm import tqdm
from PIL import Image, ImageColor
import json, os, sys
from pathlib import Path

def tensor_to_arr(tensor, ex=0):
    if len(tensor.shape) == 4:
        tensor = tensor[ex]
    return tensor.detach().permute(1, 2, 0).cpu().numpy()

def viz(tensor, ex=0):
    im = tensor_to_arr(tensor, ex)
    if im.max() > 2.0:
        im = im / 255.0
    plt.imshow(im)


def get_palette(colors_json='/home/dbear/RAFT-TDW/notebooks/colors.json', i=0):
    colors = json.loads(Path(colors_json).read_text(encoding='utf-8'))
    colors_hex = colors[i]
    colors_rgb = [ImageColor.getcolor(col, "RGB") for col in colors_hex]
    return colors_rgb

def plot_palette(i=3):
    colors = get_palette(i=i)
    arr = np.zeros((20,50,3))
    for i,c in enumerate(colors):
        arr[:,10*i:10*(i+1),:] = np.stack([np.array(c)]*10, 0) / 255.
    plt.imshow(arr)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def seg_to_rgb(seg, colors):
    size = seg.shape[:2]
    rgb = np.zeros((size[0], size[1], 3))
    for i,c in enumerate(colors):
        rgb[(seg == i),:] = c
    return rgb / 255.

def iou(pred, target):
    I = (pred & target).sum()
    U = (pred | target).sum() - I
    return (float(I) / max(U, 1))

def filter_by_iou(mask, target, thresh=0.5):
    return np.zeros_like(mask) if (iou(mask, target) > thresh) else mask

def plot_image_pred_gt_segments(data,
                                cmap='twilight',
                                bg_color=(0,0,0),
                                bg_thresh=0.5,
                                resize=None,
                                figsize=(12,4),
                                show_titles=False,
                                save_path=None,
                                do_plot=True):

    assert all((k in data.keys() for k in ('image', 'pred_segments', 'gt_segments')))

    img = data['image'].permute(1,2,0)
    if 'cuda' in str(img.device):
        img = img.cpu()
    img = img.numpy()
    size = img.shape[0:2]

    gt = np.zeros(size)
    pred = np.zeros(size)

    N, _N = len(data['pred_segments']), len(data['gt_segments'])
    assert N == _N, (N, _N)

    ## may need to resample
    if resize is None:
        R = lambda x: x
    else:
        def R(img):
            dtype = img.dtype
            img = Image.fromarray(img.astype(float))
            img = img.resize(resize, resample=Image.BILINEAR).resize(size, resample=Image.NEAREST)
            return np.array(img).astype(dtype)

    for n in range(N):
        bg = sum(data['gt_segments']) < 1
        gt += data['gt_segments'][n].astype(gt.dtype) * (n+1)
        pred += filter_by_iou(R(data['pred_segments'][n]), bg, bg_thresh).astype(pred.dtype) * (n+1)

    if isinstance(cmap, int):
        colors = get_palette(i=cmap)
        colors.insert(0, bg_color)
        gt, pred = seg_to_rgb(gt, colors), seg_to_rgb(pred, colors)


    plots = [img, pred, gt]
    titles = ['image', 'pred', 'gt']

    if do_plot:
        fig, axes = plt.subplots(1,3,figsize=figsize)
        for i,ax in enumerate(axes):
            if cmap in cmo.cmap_d.keys():
                cmap = cmo.cmap_d[cmap]
            if isinstance(cmap, int):
                ax.imshow(plots[i])
            else:
                ax.imshow(plots[i], cmap=cmap, vmin=0, vmax=(plots[i].max()+1))
            if show_titles:
                ax.set_title(titles[i])
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', format='svg', transparent=True)
        plt.show()

    out = {titles[p]:plots[p] for p in range(3)}
    out['cmap'] = cmap
    return out

def compare_models(results_dir, models, ex=0,
                   cmap='twilight', bg_color=(255,255,255), bg_thresh=0.5,
                   resize=None,
                   show_titles=True,
                   save_path=None):
    """Plot the matched segmentation results for each of the models in models List[Path]"""
    img = None
    preds, gts = {}, {}

    ## load results
    for i,m in enumerate(tqdm(models)):
        results_path = os.path.join(results_dir, m)
        results = sorted(os.listdir(results_path))
        data = torch.load(os.path.join(results_path, results[ex]))

        if i == (len(models) - 1):
            img = data['image']
        gts[m] = data['gt_segments']
        preds[m] = data['pred_segments']


    model_plots = {
        m: plot_image_pred_gt_segments(
            data={'image': img, 'gt_segments': gts[m], 'pred_segments': preds[m]},
            cmap=cmap, bg_color=bg_color, bg_thresh=bg_thresh, resize=resize, do_plot=False)
        for m in models}

    ## plot results
    _cm = model_plots[models[0]]['cmap']
    fig, axes = plt.subplots(1, len(models) + 2, figsize=(8 + 4*len(models), 4))
    def _imshow(ax, plot):
        if isinstance(cmap, int):
            ax.imshow(plot)
        else:
            ax.imshow(plot, cmap=_cm, vmin=0, vmax=(plot.max()+1))

    axes[0].imshow(model_plots[models[0]]['image'])
    _imshow(axes[-1], model_plots[models[0]]['gt'])
    if show_titles:
        axes[0].set_title('image')
        axes[-1].set_title('gt')

    for i,m in enumerate(models):
        _imshow(axes[i+1], model_plots[models[i]]['pred'])
        if show_titles:
            axes[i+1].set_title(m)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', format='svg', transparent=True)
    plt.show()
    return model_plots
