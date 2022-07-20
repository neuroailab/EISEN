from __future__ import print_function, division

import pdb
import sys

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import torch
import torch.nn as nn
from core.datasets import fetch_dataloader

from core.eisen import EISEN
from core.optimizer import fetch_optimizer
from core.raft import EvalRAFT
import core.utils.sync_batchnorm as sync_batchnorm



num_gpus = torch.cuda.device_count()
def train(args):
    total_steps = 0
    train_loader = fetch_dataloader(args)
    model = nn.DataParallel(EISEN())
    model = sync_batchnorm.convert_model(model)
    raft_model = EvalRAFT(flow_threshold=args.flow_threshold)

    optimizer, scheduler = fetch_optimizer(args, model)

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))
        total_steps = int(args.ckpt.split('_')[-1].split('.pth')[0]) + 1
        logging.info(f'Restore checkpoint from {args.ckpt}')

    model.cuda() # fixme: change it to the device of the first args.gpus
    model.train()
    logging.info(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    assert args.batch_size % num_gpus == 0, \
            f"Batch size should be divisible by #gpus, but got batch size {args.batch_size} and {num_gpus} gpus"

    if args.wandb:
        import wandb
        wandb.init(project="detectron", name=args.name, settings=wandb.Settings(start_method="fork"))

    end = time.time()
    while total_steps < args.num_steps:
        for i_batch, data_dict in enumerate(train_loader):
            data_time = time.time() - end
            optimizer.zero_grad()
            data_dict = {k: v.cuda() for k, v in data_dict.items() if k in ['img1', 'segment_target']}

            if args.compute_flow:
                with torch.no_grad():
                    _, _, segment_target = raft_model(data_dict['img1'], data_dict['img2'])
            else:
                segment_target = data_dict['segment_target']

            raft_time = time.time() - end
            _, loss, metric, segment = model(data_dict, segment_target.detach())
            loss = loss.mean()
            loss.backward()

            step_time = time.time() - end
            eta = int((args.num_steps - total_steps) * step_time)

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            scheduler.step()

            if (total_steps+ 1) % args.print_freq == 0:
                logging.info(f"[train] iter: {total_steps}  loss: {loss:.3f}  data_time: {data_time:.4f}  raft_time: {raft_time:.4f}, step_time: {step_time:.3f}  " \
                      f"eta: {str(datetime.timedelta(seconds=eta))}  " \
                      f"lr: {optimizer.param_groups[0]['lr']:.7f}")
                if args.wandb:
                    wandb.log({'loss': loss, 'data_time': data_time, 'step_time': step_time, 'lr': optimizer.param_groups[0]['lr']}, step=total_steps)

            if (total_steps + 1) % args.val_freq == 0:
                ckpt_path = f'checkpoints/{args.name}/ckpt_{total_steps}.pth'
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f'Save checkpoint to {ckpt_path}')

                # avg_miou, avg_loss = evaluate(args)
                # if args.wandb:
                #     wandb.log({'val_loss': avg_loss, 'val_miou': avg_miou}, step=total_steps)
                #
                # model.train()
            total_steps += 1
            end = time.time()

    # saving final checkpoint
    torch.save(model.state_dict(), f'checkpoints/{args.name}/ckpt_final.pth')


def evaluate(args):
    val_loader = fetch_dataloader(args, training=False, drop_last=False)
    model = nn.DataParallel(EISEN())
    model = sync_batchnorm.convert_model(model)
    raft_model = EvalRAFT(flow_threshold=args.flow_threshold) if args.compute_flow else None

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))
        logging.info(f'Restore checkpoint from {args.ckpt}')
    else:
        logging.warning('Warning: continue evaluation without loading pretrained checkpoints')

    model.cuda()
    logging.info(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    assert args.batch_size / num_gpus == 1, \
            f"Effective Batch size should be 1, but got batch size {args.batch_size} and {num_gpus} gpus"

    avg_miou, avg_loss = evaluate_helper(args, val_loader, model, raft_model=raft_model)
    return avg_miou, avg_loss


def evaluate_helper(args, dataloader, model, raft_model=None):
    # print('!!!!!!!!!!!! warning: loading checkpoint' * 20)
    # model.module.load_checkpoint()

    model = model.cuda()
    model.eval()
    end = time.time()
    miou_list = []
    loss_list = []
    time_list = []

    for step, data_dict in enumerate(dataloader):
        start = time.time()
        data_time = time.time() - end
        data_dict = {k: v.cuda() for k, v in data_dict.items() if not k == 'file_name'}

        bs = data_dict['img1'].shape[0]
        if bs < args.batch_size:  # add padding if bs is smaller than batch size
            pad_size = args.batch_size - bs

            for key in data_dict.keys():
                padding = torch.cat([torch.zeros_like(data_dict[key][0:1])] * pad_size, dim=0)
                data_dict[key] = torch.cat([data_dict[key], padding], dim=0)

        if args.compute_flow:
            with torch.no_grad():
                _, _, segment_target = raft_model(data_dict['img1'], data_dict['img2'])
        else:
            segment_target = data_dict['segment_target']

        raft_time = time.time() - end
        _, loss, metric, segment = model(data_dict, segment_target.detach(), get_segments=True)

        mious = metric['metric_pred_segment_mean_ious']

        step_time = time.time() - end
        eta = int((len(dataloader) - step) * step_time)

        if num_gpus == 1:
            loss_list.append(loss.item())
            miou_list.append(mious.item())
        else:
            loss_list.extend([l.item() for l in list(loss)][0:bs])
            miou_list.extend([miou.item() for miou in list(mious)][0:bs])

        avg_miou = np.nanmean(miou_list)
        avg_loss = np.nanmean(loss_list)
        logging.info(f"[val] iter: {step} avg_miou: {avg_miou:.3f} avg_loss: {avg_loss:.3f}  " \
                     f"data_time: {data_time:.4f}  raft_time: {raft_time:.3f} step_time: {step_time:.3f}  " \
                     f"eta: {str(datetime.timedelta(seconds=eta))}")
        end = time.time()
        time_list.append(end-start)

    avg_time = np.mean(time_list[5:-1]) # the first 5 iterations are warmup

    print(f'Num. imgs: {len(miou_list)}')
    print(f'Avg. mIoU: {avg_miou:.3f}')
    print(f'Avg. time: {avg_time:.3f}')
    return avg_miou, avg_loss

def precompute_flows(args):
    answer = input("Continue to pre-compute optical flows (y/n): ")
    if not answer == "y":
        exit()

    total_steps = start_step = 0
    train_loader = fetch_dataloader(args, drop_last=False)
    train_loader = iter(train_loader)
    raft_model = EvalRAFT(flow_threshold=args.flow_threshold)

    assert args.batch_size % num_gpus == 0, \
            f"Batch size should be divisible by #gpus, but got batch size {args.batch_size} and {num_gpus} gpus"
    end = time.time()

    for step in range(start_step, len(train_loader)):

        data_dict = next(train_loader)
        data_time = time.time() - end
        file_name = data_dict['file_name']
        data_dict = {k: v.cuda() for k, v in data_dict.items() if not k == 'file_name'}

        bs = data_dict['img1'].shape[0]
        if bs < args.batch_size: # add padding if bs is smaller than batch size
            pad_size = args.batch_size - bs
            padding = torch.zeros_like(data_dict['img1'][0:1]).expand(pad_size, -1, -1, -1)
            for key in ['img1', 'img2']:
                data_dict[key] = torch.cat([data_dict[key], padding], dim=0)

        with torch.no_grad():
            flow, magnitude, segment_target = raft_model(data_dict['img1'], data_dict['img2'])

            for i, name in enumerate(file_name):
                if i < bs:
                    flow_uv = flow[i].cpu().numpy()
                    image_name = name.split('/')[-1]
                    np.save(name.replace(image_name, 'flow_'+image_name).replace('png', 'npy'), flow_uv)

        step_time = time.time() - end
        eta = int((len(train_loader) - step) * step_time)

        logging.info(f"iter: {step}  data_time: {data_time:.4f} step_time: {step_time:.3f}  " \
              f"eta: {str(datetime.timedelta(seconds=eta))}")

        total_steps += 1
        end = time.time()


def get_model_args(args):
    params = dict()
    params['affinity_res'] = args.affinity_size
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--name', default='eisen', help="name your experiment")
    parser.add_argument('--dataset', default="playroom", help="determines which dataset to use for training")

    # dataloader
    parser.add_argument('--num_workers', type=int, default=8)

    # training
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--clip_grad', type=float, default=1.0, help='gradient clipping value')
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--affinity_size', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--flow_threshold', type=float, default=0.5, help='binary threshold on raft dlows')
    parser.add_argument('--ckpt', type=str, help='path to restored checkpoint')

    # evaluation
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--val_freq', type=int, default=5000, help='validation and checkpoint frequency')

    # logging
    parser.add_argument('--print_freq', type=int, default=100, help='frequency for printing loss')
    parser.add_argument('--wandb', action='store_true', help='enable wandb login')

    # flow
    parser.add_argument('--precompute_flow', action='store_true', help='compute flow before training (recommended)')
    parser.add_argument('--compute_flow', action='store_true', help='compute flow during training (slower option)')

    args = parser.parse_args()
    torch.manual_seed(1)
    np.random.seed(1)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    if not os.path.isdir(f'checkpoints/{args.name}'):
        os.mkdir(f'checkpoints/{args.name}')

    if args.precompute_flow:
        precompute_flows(args)
    elif args.eval_only:
        evaluate(args)
    else:
        train(args)
