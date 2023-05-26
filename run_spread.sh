#!/bin/bash
python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 


python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.3
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.3

python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.5
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.5


python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.7
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.7


python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --N 5
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --N 5


python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.3 --N 5
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.3 --N 5

python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.5 --N 5
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.5 --N 5

python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.7 --N 5
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.7 --N 5



python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --N 10
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --N 10

python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.3 --N 10
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.3 --N 10

python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.5 --N 10
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.5 --N 10

python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.7 --N 10
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.7 --N 10


python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --N 20
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --N 20

python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.3 --N 20
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.3 --N 20

python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.5 --N 20
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.5 --N 20

python spread.py -m gnn  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.7 --N 20
python spread.py -m mlp  --train_batch_size 1024 --lr 0.0001 --use_wandb --num_workers 3 --mask_prob 0.7 --N 20




























