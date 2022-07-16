from __future__ import print_function, division

import pdb
import sys

from core.datasets import fetch_dataloader

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import torch
import torch.nn as nn
from core.eisen import EISEN
from core.optimizer import fetch_optimizer
from core.raft import EvalRAFT
import datetime



def train(args):
    total_steps = 0
    train_loader = fetch_dataloader(args)
    model = nn.DataParallel(EISEN(), device_ids=args.gpus)
    raft_model = EvalRAFT(device_ids=args.gpus, flow_threshold=args.flow_threshold)

    optimizer, scheduler = fetch_optimizer(args, model)

    if args.restore_ckpt is not None:
        state_dict = torch.load(args.restore_ckpt)
        model.load_state_dict(state_dict, strict=False)
        total_steps = state_dict['total_steps']
        logging.info('Restore checkpoint from ', args.restore_ckpt)

    model.cuda()
    model.train()
    logging.info(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    assert args.batch_size % len(args.gpus) == 0, \
            f"Batch size should be divisible by #gpus, but got batch size {args.batch_size} and {args.gpus} gpus"

    if args.wandb:
        import wandb
        wandb.init(project="detectron", name=args.name, settings=wandb.Settings(start_method="fork"))

    end = time.time()
    while total_steps < args.num_steps:
        for i_batch, data_dict in enumerate(train_loader):
            data_time = time.time() - end
            optimizer.zero_grad()
            data_dict = {k: v.cuda() for k, v in data_dict.items()}

            if args.compute_flow:
                with torch.no_grad():
                    _, _, segment_target = raft_model(data_dict['img1'], data_dict['img2'])
            else:
                segment_target = data_dict['segment_target']

            raft_time = time.time() - end
            _, loss, _ = model(data_dict['img1'], segment_target.detach())
            loss = loss.mean()
            loss.backward()

            step_time = time.time() - end
            eta = int((args.num_steps - total_steps) * step_time)

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            scheduler.step()

            if (total_steps+ 1) % args.print_freq == 0:
                logging.info(f"iter: {total_steps}  loss: {loss:.3f}  data_time: {data_time:.4f}  raft_time: {raft_time:.4f}, step_time: {step_time:.3f}  " \
                      f"eta: {str(datetime.timedelta(seconds=eta))}  " \
                      f"lr: {optimizer.param_groups[0]['lr']:.7f}")
                if args.wandb:
                    wandb.log({'loss': loss, 'data_time': data_time, 'step_time': step_time, 'lr': optimizer.param_groups[0]['lr']}, step=total_steps)

            if (total_steps + 1) % args.val_freq == 0:
                torch.save(model.state_dict(), f'checkpoints/{args.name}/ckpt_{total_steps}.pth')
                model.train()
            total_steps += 1
            end = time.time()

    # saving final checkpoint
    torch.save(model.state_dict(), f'checkpoints/{args.name}/ckpt_final.pth')


def validate(dataloader, model):
    end = time.time()
    for step in range(len(dataloader)):
        data_dict = next(dataloader)

        data_time = time.time() - end
        data_dict = {k: v.cuda() for k, v in data_dict.items() if not k == 'file_name'}

        # with torch.no_grad():
        #     _, _, segment_target = raft_model(data_dict['img1'], data_dict['img2'])
        segment_target = data_dict['segment_target']

        raft_time = time.time() - end
        _, loss, _ = model(data_dict['img1'], segment_target.detach(), get_segment=True)
        loss = loss.mean()

        step_time = time.time() - end
        eta = int((args.num_steps - step) * step_time)

        if (step+ 1) % args.print_freq == 0:
            logging.info(f"iter: {step}  loss: {loss:.3f}  data_time: {data_time:.4f}  raft_time: {raft_time:.4f}, step_time: {step_time:.3f}  " \
                  f"eta: {str(datetime.timedelta(seconds=eta))}")


def precompute_flows(args):
    answer = input("Continue to pre-compute optical flows (y/n): ")
    if not answer == "y":
        exit()

    total_steps = start_step = 0
    train_loader = fetch_dataloader(args, drop_last=False)
    train_loader = iter(train_loader)
    raft_model = EvalRAFT(device_ids=args.gpus, flow_threshold=args.flow_threshold)

    assert args.batch_size % len(args.gpus) == 0, \
            f"Batch size should be divisible by #gpus, but got batch size {args.batch_size} and {args.gpus} gpus"
    end = time.time()

    for step in range(start_step, len(train_loader)):
        data_dict = next(train_loader)
        data_time = time.time() - end
        file_name = data_dict['file_name']
        data_dict = {k: v.cuda() for k, v in data_dict.items() if not k == 'file_name'}

        with torch.no_grad():
            flow, magnitude, segment_target = raft_model(data_dict['img1'], data_dict['img2'])

            for i, name in enumerate(file_name):
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
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--flow_threshold', type=float, default=0.5, help='binary threshold on raft dlows')
    parser.add_argument('--restore_ckpt', type=str, help='path to restored checkpoint')

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

    if args.eval_only:
        eval(args)
    elif args.precompute_flow:
        precompute_flows(args)
    else:
        train(args)
