#!/bin/bash
mkdir -p checkpoints
# 4.1.22
#python -u train_eisen.py --name eisen_unsup --stage tdw_png --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0005 --image_size 512 512 --wdecay 0.0001 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5
python -u train_eisen.py --name eisen_raft_0.5 --teacher_class raft_pretrained --stage tdw_png --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0005 --image_size 512 512 --wdecay 0.0001 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5
python -u train_eisen.py --name eisen_unsup_improved --stage tdw_png --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0005 --image_size 512 512 --wdecay 0.0001 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5

python -u train_eisen.py --name eisen_raft_0.5_bs2 --teacher_class raft_pretrained --stage tdw_png --gpus 0 --num_steps 100000 --batch_size 2 --lr 0.0005 --image_size 512 512 --wdecay 0.000 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5

python -u train_eisen.py --name eisen_teacher_v1_bs4 --teacher_class motion_to_static_v1 --stage movi_d --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0005 --image_size 512 512 --wdecay 0.000 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5

# 4b6e77dfe907889d46352de43f48dbd8bdfad1ec
python -u train_eisen.py --name eisen_teacher_v1_bs4_improve-5-5 --teacher_class motion_to_static_v1 --stage movi_d --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0005 --image_size 512 512 --wdecay 0.000 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5

# eae3ab7
python -u train_eisen.py --name eisen_teacher_v1_64_bs16 --teacher_class motion_to_static_v1 --stage movi_d --gpus 0 --num_steps 100000 --batch_size 16 --lr 0.0005 --image_size 512 512 --wdecay 0.000 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5
python -u train_eisen.py --name eisen_teacher_v1_128_bs4 --stem_pool 0 --teacher_class motion_to_static_v1 --stage movi_d --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0005 --image_size 512 512 --wdecay 0.000 --no_aug --full_playroom --filepattern *[0-8] --max_frame 5