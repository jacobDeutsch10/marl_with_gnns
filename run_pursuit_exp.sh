#!/bin/bash

python pursuit.py -m mlp  --mask_prob 0.3 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb
python pursuit.py -m gat  --mask_prob 0.3 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb

python pursuit.py -m mlp  --mask_prob 0.5 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb
python pursuit.py -m gat  --mask_prob 0.5 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb


python pursuit.py -m mlp  --mask_prob 0.7 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb
python pursuit.py -m gat  --mask_prob 0.7 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb

python pursuit.py -m mlp  --env_size 32 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb
python pursuit.py -m gat  --env_size 32 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb


python pursuit.py -m mlp  --obs_range 16 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb
python pursuit.py -m gat  --obs_range 16 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb


python pursuit.py -m mlp  --env_size 32 --obs_range 16 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb
python pursuit.py -m gat  --env_size 32 --obs_range 16 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb


python pursuit.py -m mlp  --n_evaders 60 --n_pursuers 8 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb
python pursuit.py -m gat  --n_evaders 60 --n_pursuers 8 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb

python pursuit.py -m mlp  --n_evaders 60 --n_pursuers 8 --env_size 32 --obs_range 16 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb
python pursuit.py -m gat  --n_evaders 60 --n_pursuers 8 --env_size 32 --obs_range 16 --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb

python pursuit.py -m mlp  --n_evaders 60 --n_pursuers 8 --env_size 32  --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb
python pursuit.py -m gat  --n_evaders 60 --n_pursuers 8 --env_size 32  --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes --num_workers 3 --use_wandb











