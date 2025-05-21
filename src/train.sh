#!/bin/bash

# --- Transformer --- #
#python3 src/train.py --task_type classification --labeling_strategy tercile --output_dim 3 --model_name transformer --d_model 512 --nhead 8 --num_layers 2 --dim_ff 512 --dropout 0.3 --epochs 20 --batch_size 1024 --lr 0.0001 --weight_decay 0.001 --early_stopping_patience 10 --num_workers 4 --seed 42

python3 src/train.py --task_type classification --labeling_strategy tercile --output_dim 3 --model_name transformer --d_model 512 --nhead 16 --num_layers 6 --dim_ff 2048 --dropout 0.5 --epochs 150 --batch_size 1024 --lr 0.00001 --weight_decay 0.01 --early_stopping_patience 50 --num_workers 4 --seed 42


# --- LSTM --- #
#python3 src/train.py --task_type classification --labeling_strategy tercile --output_dim 3 --model_name lstm --hidden_dim 512 --num_layers 2 --dropout 0.5 --epochs 25 --batch_size 1024 --lr 0.0001 --weight_decay 0.001 --early_stopping_patience 30 --num_workers 4 --seed 42

# --- TCN --- #
#python3 src/train.py --task_type classification --labeling_strategy tercile --output_dim 3 --model_name tcn --hidden_dim 512 --num_layers 2 --dropout 0.1 --epochs 5 --batch_size 1024 --lr 0.001 --weight_decay 0.01 --early_stopping_patience 30 --num_workers 4 --seed 42


# --- SLURM Example --- # d
# srun --gres=gpu:1 bash src/train.sh
